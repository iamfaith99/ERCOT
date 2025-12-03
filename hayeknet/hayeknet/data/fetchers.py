"""Data fetching methods for ERCOT APIs."""
from __future__ import annotations

import zipfile
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    from pyiso import ERCOT  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    ERCOT = None  # pyiso is optional in early development

from hayeknet.data.endpoints import ERCOTEndpoints, ERCOTSchemaMapping


def estimate_lmp_from_load(load_mw: float) -> float:
    """Estimate LMP in USD/MWh from system load using a simple affine model."""
    baseline = 25.0
    slope = 0.001
    estimated = baseline + slope * (load_mw - 45000.0)
    return float(max(10.0, min(estimated, 400.0)))


def fetch_ercot_dashboard_data(
    endpoints: ERCOTEndpoints,
    timeout: int = 30
) -> pd.DataFrame:
    """Fetch real-time data from working ERCOT dashboard APIs."""
    try:
        # Working endpoints discovered
        working_endpoints = {
            'fuel_mix': 'https://www.ercot.com/api/1/services/read/dashboards/fuel-mix',
            'supply_demand': 'https://www.ercot.com/api/1/services/read/dashboards/supply-demand',
            'system_demand': 'https://www.ercot.com/api/1/services/read/dashboards/system-wide-demand'
        }

        current_time = datetime.utcnow()

        # Accumulate dataframes from multiple dashboard endpoints
        collected_frames: list[pd.DataFrame] = []

        # Try to get supply-demand data (most promising)
        try:
            response = requests.get(working_endpoints['supply_demand'], timeout=timeout)
            response.raise_for_status()

            if 'json' in response.headers.get('content-type', '').lower():
                payload = response.json()

                if isinstance(payload, dict):
                    records: list[dict[str, object]] = []

                    realtime_block = payload.get('data', [])
                    if isinstance(realtime_block, list):
                        for entry in realtime_block:
                            if not isinstance(entry, dict):
                                continue
                            timestamp = pd.to_datetime(entry.get('timestamp'), utc=True, errors='coerce')
                            demand = entry.get('demand') or entry.get('currentLoadForecast')
                            available = entry.get('available') or entry.get('availableCapacity') or entry.get('availableCap')
                            if timestamp is None or demand is None:
                                continue
                            try:
                                load_mw = float(demand)
                            except (TypeError, ValueError):
                                continue

                            records.append({
                                'timestamp': timestamp,
                                'net_load_mw': load_mw,
                                'available_capacity_mw': float(available) if available is not None else np.nan,
                                'is_forecast': bool(entry.get('forecast', 0)),
                                'interval_minutes': int(entry.get('interval', 0)) if entry.get('interval') is not None else None,
                                'data_source': 'ercot_dashboard_realtime'
                            })

                    forecast_block = payload.get('forecast', [])
                    if isinstance(forecast_block, list):
                        for entry in forecast_block:
                            if not isinstance(entry, dict):
                                continue
                            timestamp = pd.to_datetime(entry.get('timestamp') or entry.get('deliveryDateHrBegin'), utc=True, errors='coerce')
                            demand = entry.get('forecastedDemand') or entry.get('forecastDemand')
                            available = entry.get('availCapGen')
                            if timestamp is None or demand is None:
                                continue
                            try:
                                load_mw = float(demand)
                            except (TypeError, ValueError):
                                continue

                            records.append({
                                'timestamp': timestamp,
                                'net_load_mw': load_mw,
                                'available_capacity_mw': float(available) if available is not None else np.nan,
                                'is_forecast': True,
                                'interval_minutes': 60,
                                'data_source': 'ercot_dashboard_forecast'
                            })

                    if records:
                        supply_df = pd.DataFrame(records).dropna(subset=['timestamp'])
                        supply_df = supply_df.sort_values('timestamp').reset_index(drop=True)
                        supply_df['lmp_usd'] = supply_df['net_load_mw'].apply(estimate_lmp_from_load)
                        supply_df['ingested_at'] = pd.Timestamp(current_time, tz='UTC')
                        collected_frames.append(supply_df)
                        print(f"✅ Supply-demand dashboard frame: {len(supply_df)} records")

        except Exception as e:
            print(f"Warning: Supply-demand endpoint failed: {e}")

        # Try to get fuel mix data (provides generation by source)
        try:
            response = requests.get(working_endpoints['fuel_mix'], timeout=timeout)
            response.raise_for_status()

            if 'json' in response.headers.get('content-type', '').lower():
                payload = response.json()

                mix_records: list[dict[str, object]] = []
                if isinstance(payload, dict):
                    data_block = payload.get('data') or payload.get('series') or []
                    if isinstance(data_block, list):
                        for entry in data_block:
                            if not isinstance(entry, dict):
                                continue
                            timestamp = pd.to_datetime(entry.get('timestamp'), utc=True, errors='coerce')
                            if timestamp is None:
                                continue

                            mix_entry = {
                                'timestamp': timestamp,
                                'nuclear_mw': entry.get('nuclear'),
                                'coal_mw': entry.get('coal'),
                                'gas_mw': entry.get('gas'),
                                'wind_mw': entry.get('wind'),
                                'solar_mw': entry.get('solar'),
                                'hydro_mw': entry.get('hydro'),
                                'other_mw': entry.get('other'),
                                'total_generation_mw': entry.get('total'),
                                'data_source': 'ercot_dashboard_fuel_mix'
                            }

                            mix_records.append(mix_entry)

                if mix_records:
                    fuel_df = pd.DataFrame(mix_records).dropna(subset=['timestamp'])
                    fuel_df = fuel_df.sort_values('timestamp').reset_index(drop=True)
                    fuel_df['ingested_at'] = pd.Timestamp(current_time, tz='UTC')
                    collected_frames.append(fuel_df)
                    print(f"✅ Fuel-mix dashboard frame: {len(fuel_df)} records")

        except Exception as e:
            print(f"Warning: Fuel-mix endpoint failed: {e}")

        # Try system-wide-demand endpoint as a fallback (still single-point)
        try:
            response = requests.get(working_endpoints['system_demand'], timeout=timeout)
            response.raise_for_status()

            if 'json' in response.headers.get('content-type', '').lower():
                data = response.json()

                load_mw = None

                if isinstance(data, (list, dict)):
                    if isinstance(data, list) and data:
                        item = data[0] if isinstance(data[0], dict) else {}
                    else:
                        item = data if isinstance(data, dict) else {}

                    for key, value in item.items():
                        key_lower = key.lower() if isinstance(key, str) else ''
                        if any(term in key_lower for term in ['demand', 'load', 'mw']):
                            try:
                                load_mw = float(value)
                                break
                            except (ValueError, TypeError):
                                continue

                if load_mw is not None:
                    lmp_price = estimate_lmp_from_load(load_mw)
                    sys_df = pd.DataFrame({
                        'timestamp': [pd.Timestamp(current_time, tz='UTC')],
                        'net_load_mw': [load_mw],
                        'available_capacity_mw': [np.nan],
                        'is_forecast': [False],
                        'interval_minutes': [None],
                        'data_source': ['ercot_dashboard_system_demand'],
                        'lmp_usd': [lmp_price],
                        'ingested_at': [pd.Timestamp(current_time, tz='UTC')]
                    })

                    collected_frames.append(sys_df)
                    print(f"✅ Real ERCOT system demand: {load_mw:.0f} MW, ${lmp_price:.2f}/MWh (estimated)")

        except Exception as e:
            print(f"Warning: System demand endpoint failed: {e}")

        if collected_frames:
            combined = pd.concat(collected_frames, ignore_index=True, sort=False)
            combined = combined.sort_values('timestamp').reset_index(drop=True)
            return combined

        return pd.DataFrame()
            
    except Exception as e:
        print(f"Warning: Dashboard data fetch failed: {e}")
        return pd.DataFrame()


def fetch_lmp_data(
    endpoints: ERCOTEndpoints,
    schema: ERCOTSchemaMapping,
    lookback_hours: int,
    timeout: int,
    *,
    latest: bool
) -> pd.DataFrame:
    """Fetch LMP data from NP6-788-CD endpoint with enhanced error handling."""
    end_date = datetime.now().date() if latest else (datetime.now() - timedelta(hours=lookback_hours)).date()
    
    # ERCOT Data Portal API call
    url = f"{endpoints.data_portal_base}datasets/{endpoints.real_time_lmp}"
    params = {
        "deliveryDateFrom": (end_date - timedelta(days=1)).isoformat(),
        "deliveryDateTo": end_date.isoformat(),
        "settlementPointType": "HUB",  # Focus on major hubs
        "format": "csv"
    }
    
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        
        # Check if response is actually CSV data
        content_type = response.headers.get('content-type', '').lower()
        
        if 'html' in content_type:
            print(f"Warning: ERCOT API returned HTML instead of CSV (endpoint may require registration)")
            return pd.DataFrame()
        
        if 'xml' in content_type:
            print(f"Warning: ERCOT API returned XML instead of CSV")
            return pd.DataFrame()
        
        # Handle potential ZIP response
        if content_type.startswith('application/zip') or response.content.startswith(b'PK'):
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                csv_file = next((name for name in z.namelist() if name.endswith('.csv')), None)
                if csv_file:
                    df = pd.read_csv(z.open(csv_file))
                else:
                    return pd.DataFrame()
        else:
            # Check if content looks like CSV
            text_content = response.text.strip()
            if not text_content or text_content.startswith('<!DOCTYPE') or text_content.startswith('<html'):
                print(f"Warning: ERCOT API returned HTML page instead of data")
                return pd.DataFrame()
            
            df = pd.read_csv(BytesIO(response.content))
        
        # Apply schema mapping
        df = df.rename(columns=schema.lmp_schema)
        
        # Create timestamp from date/hour/interval
        df['timestamp'] = pd.to_datetime(df['delivery_date']) + \
                        pd.to_timedelta((df['delivery_hour'] - 1) * 60 + (df['delivery_interval'] - 1) * 5, unit='min')
        
        # Average LMPs across hubs for system-wide price
        system_lmp = df.groupby('timestamp')['lmp_usd'].mean().reset_index()
        
        return system_lmp
        
    except Exception as e:
        print(f"Warning: LMP data fetch failed: {e}")
        return pd.DataFrame()


def fetch_load_data(
    endpoints: ERCOTEndpoints,
    schema: ERCOTSchemaMapping,
    lookback_hours: int,
    timeout: int,
    *,
    latest: bool
) -> pd.DataFrame:
    """Fetch load data from NP3-910-ER endpoint with enhanced error handling."""
    end_date = datetime.now().date() if latest else (datetime.now() - timedelta(hours=lookback_hours)).date()
    
    url = f"{endpoints.data_portal_base}datasets/{endpoints.real_time_load}"
    params = {
        "operDayFrom": (end_date - timedelta(days=1)).isoformat(),
        "operDayTo": end_date.isoformat(),
        "format": "csv"
    }
    
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        
        # Check if response is actually CSV data
        content_type = response.headers.get('content-type', '').lower()
        
        if 'html' in content_type:
            print(f"Warning: ERCOT API returned HTML instead of CSV (endpoint may require registration)")
            return pd.DataFrame()
        
        if 'xml' in content_type:
            print(f"Warning: ERCOT API returned XML instead of CSV")
            return pd.DataFrame()
        
        # Handle potential ZIP response
        if content_type.startswith('application/zip') or response.content.startswith(b'PK'):
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                csv_file = next((name for name in z.namelist() if name.endswith('.csv')), None)
                if csv_file:
                    df = pd.read_csv(z.open(csv_file))
                else:
                    return pd.DataFrame()
        else:
            # Check if content looks like CSV
            text_content = response.text.strip()
            if not text_content or text_content.startswith('<!DOCTYPE') or text_content.startswith('<html'):
                print(f"Warning: ERCOT API returned HTML page instead of data")
                return pd.DataFrame()
            
            df = pd.read_csv(BytesIO(response.content))
        
        # Apply schema mapping
        df = df.rename(columns=schema.load_schema)
        
        # Create timestamp
        df['timestamp'] = pd.to_datetime(df['oper_day']) + pd.to_timedelta(df['hour_ending'] - 1, unit='h')
        
        # Use total ERCOT load
        load_data = df[['timestamp', 'total_load_mw']].copy()
        load_data = load_data.rename(columns={'total_load_mw': 'net_load_mw'})
        
        # Interpolate hourly to 5-minute intervals
        load_data = load_data.set_index('timestamp').resample('5T').interpolate().reset_index()
        
        return load_data
        
    except Exception as e:
        print(f"Warning: Load data fetch failed: {e}")
        return pd.DataFrame()


def fetch_pyiso_data(
    lookback_hours: int,
    *,
    latest: bool
) -> pd.DataFrame:
    """Fallback to pyiso client (legacy method)."""
    if ERCOT is None:
        return pd.DataFrame()
    
    client = ERCOT()
    end_time = datetime.utcnow() if latest else datetime.utcnow() - timedelta(hours=lookback_hours)
    start_time = end_time - timedelta(hours=lookback_hours)

    try:
        load = client.get_load(start_at=start_time, end_at=end_time)
        lmp = client.get_lmp(start_at=start_time, end_at=end_time)
    except Exception:
        return pd.DataFrame()

    load_df = pd.DataFrame(load)
    lmp_df = pd.DataFrame(lmp)

    if load_df.empty or lmp_df.empty:
        return pd.DataFrame()

    merged = pd.merge_asof(
        load_df.sort_values("timestamp"),
        lmp_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )
    merged = merged.rename(columns={"load": "net_load_mw", "price": "lmp_usd"})
    return merged[["timestamp", "net_load_mw", "lmp_usd"]]


def generate_synthetic_frame(lookback_hours: int) -> pd.DataFrame:
    """Generate synthetic ERCOT-like data for testing."""
    horizon = lookback_hours * 12  # 5-minute intervals
    base_time = datetime.utcnow() - timedelta(hours=lookback_hours)
    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(horizon)]
    
    # More realistic ERCOT load patterns (45-75 GW range)
    daily_pattern = np.sin(np.linspace(0, 2 * np.pi, horizon))  # Daily cycle
    weekly_pattern = 0.3 * np.sin(np.linspace(0, 2 * np.pi / 7, horizon))  # Weekly cycle
    noise = np.random.normal(0, 2000, horizon)  # Random variations
    
    net_load = 60_000 + 10_000 * daily_pattern + 5_000 * weekly_pattern + noise
    
    # LMP patterns correlated with load (higher load = higher prices)
    base_lmp = 30 + 20 * (net_load - 50_000) / 20_000  # Load-price correlation
    lmp_noise = np.random.normal(0, 5, horizon)
    lmp = np.maximum(base_lmp + lmp_noise, 0)  # Prices can't be negative
    
    return pd.DataFrame({
        "timestamp": timestamps, 
        "net_load_mw": net_load, 
        "lmp_usd": lmp
    })


def fetch_ancillary_from_api(
    endpoints: ERCOTEndpoints,
    schema: ERCOTSchemaMapping,
    lookback_hours: int,
    timeout: int,
    *,
    latest: bool = True
) -> pd.DataFrame:
    """Try direct API fetch for ancillary service prices."""
    end_date = datetime.now().date() if latest else (datetime.now() - timedelta(hours=lookback_hours)).date()
    
    url = f"{endpoints.data_portal_base}datasets/{endpoints.ancillary_prices}"
    params = {
        "deliveryDateFrom": (end_date - timedelta(days=1)).isoformat(),
        "deliveryDateTo": end_date.isoformat(),
        "format": "csv"
    }
    
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        
        # Check if response is HTML (API error)
        content_type = response.headers.get('content-type', '')
        if 'html' in content_type.lower() or response.text.strip().startswith('<!DOCTYPE'):
            print(f"Warning: API returned HTML instead of data")
            return pd.DataFrame()
        
        if response.headers.get('content-type', '').startswith('application/zip'):
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                csv_file = next((name for name in zf.namelist() if name.endswith('.csv')), None)
                if csv_file:
                    with zf.open(csv_file) as f:
                        df = pd.read_csv(f)
                else:
                    return pd.DataFrame()
        else:
            df = pd.read_csv(BytesIO(response.content))
        
        # Apply schema mapping
        df = df.rename(columns=schema.ancillary_schema)
        
        # Create timestamp
        if 'delivery_date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['delivery_date'])
            if 'hour_ending' in df.columns:
                df['timestamp'] += pd.to_timedelta(df['hour_ending'] - 1, unit='h')
        
        return df[['timestamp', 'ancillary_type', 'market_clearing_price']] if 'ancillary_type' in df.columns else pd.DataFrame()
        
    except Exception as e:
        print(f"Warning: Direct API AS prices fetch failed: {e}")
        return pd.DataFrame()

