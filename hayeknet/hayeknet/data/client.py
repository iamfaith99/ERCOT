"""Main ERCOT data client."""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from hayeknet.data.endpoints import ERCOTEndpoints, ERCOTSchemaMapping
from hayeknet.data.fetchers import (
    fetch_ercot_dashboard_data,
    fetch_lmp_data,
    fetch_load_data,
    fetch_pyiso_data,
    generate_synthetic_frame,
    fetch_ancillary_from_api,
)
from hayeknet.data.processors import (
    build_observation_operator,
    derive_as_prices_from_lmp,
)

try:
    from pyiso import ERCOT  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    ERCOT = None  # pyiso is optional in early development


@dataclass
class ERCOTDataClient:
    """Enhanced ERCOT data client with real API endpoints and schema mapping."""

    lookback_hours: int = 24
    endpoints: ERCOTEndpoints = field(default_factory=ERCOTEndpoints)
    schema: ERCOTSchemaMapping = field(default_factory=ERCOTSchemaMapping)
    timeout: int = 30
    max_retries: int = 3
    config_path: Optional[Path] = None

    def __post_init__(self):
        """Load configuration from file if provided."""
        if self.config_path is None:
            # Default to config file in project root
            # hayeknet/hayeknet/data/client.py -> hayeknet/ -> config/
            project_root = Path(__file__).resolve().parents[2]
            self.config_path = project_root / "config" / "ercot_config.json"
        
        # Set up data cache directory
        self.cache_dir = project_root / "data" / "raw" / "ercot_historical"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config_path.exists():
            self._load_config()

    def _load_config(self):
        """Load ERCOT configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Update endpoints from config
            if 'endpoints' in config:
                for key, value in config['endpoints'].items():
                    if hasattr(self.endpoints, key):
                        setattr(self.endpoints, key, value)
            
            # Update data product IDs
            if 'data_products' in config:
                for product, details in config['data_products'].items():
                    attr_name = product
                    if hasattr(self.endpoints, attr_name) and 'emil_id' in details:
                        setattr(self.endpoints, attr_name, details['emil_id'])
            
            # Update schema mappings
            if 'schema_mappings' in config:
                schema_config = config['schema_mappings']
                if 'lmp_fields' in schema_config:
                    self.schema.lmp_schema.update(schema_config['lmp_fields'])
                if 'load_fields' in schema_config:
                    self.schema.load_schema.update(schema_config['load_fields'])
                if 'ancillary_fields' in schema_config:
                    self.schema.ancillary_schema.update(schema_config['ancillary_fields'])
            
            # Update API settings
            if 'api_settings' in config:
                settings = config['api_settings']
                self.timeout = settings.get('timeout_seconds', self.timeout)
                self.max_retries = settings.get('max_retries', self.max_retries)
                self.lookback_hours = settings.get('default_lookback_hours', self.lookback_hours)
                
        except Exception as e:
            print(f"Warning: Could not load ERCOT config from {self.config_path}: {e}")
            print("Using default configuration.")

    def fetch_rtc_like(self, *, latest: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """Return a dataframe plus observation matrix for data assimilation."""
        # Try real ERCOT data first, fallback to pyiso, then synthetic
        df = self._fetch_real_ercot_data(latest=latest)
        
        if df.empty and ERCOT is not None:
            df = fetch_pyiso_data(self.lookback_hours, latest=latest)
            
        if df.empty:
            df = generate_synthetic_frame(self.lookback_hours)

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["ancillary_shadow"] = 0.05 * df["net_load_mw"] + np.random.normal(0, 10, len(df))
        obs_matrix = df[["net_load_mw", "ancillary_shadow"]].to_numpy(dtype=float).T
        return df, obs_matrix

    def _fetch_real_ercot_data(self, *, latest: bool) -> pd.DataFrame:
        """Fetch data from working ERCOT Dashboard APIs."""
        try:
            # Try working ERCOT dashboard endpoints first
            dashboard_df = fetch_ercot_dashboard_data(self.endpoints, self.timeout)
            if not dashboard_df.empty:
                return dashboard_df
            
            # Fallback to original data portal approach
            lmp_df = fetch_lmp_data(
                self.endpoints, self.schema, self.lookback_hours, self.timeout, latest=latest
            )
            load_df = fetch_load_data(
                self.endpoints, self.schema, self.lookback_hours, self.timeout, latest=latest
            )
            
            if lmp_df.empty or load_df.empty:
                return pd.DataFrame()
                
            # Merge on timestamp with 5-minute tolerance
            merged = pd.merge_asof(
                lmp_df.sort_values("timestamp"),
                load_df.sort_values("timestamp"), 
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(minutes=5)
            )
            
            return merged[["timestamp", "net_load_mw", "lmp_usd"]].dropna()
            
        except Exception as e:
            print(f"Warning: Failed to fetch real ERCOT data: {e}")
            return pd.DataFrame()

    def fetch_ancillary_prices(self, *, latest: bool = True) -> pd.DataFrame:
        """Fetch ancillary service prices using multiple approaches."""
        print("🔍 Fetching ancillary service prices...")
        
        # Approach 1: Try historical reports (most reliable)
        df = self._fetch_ancillary_from_reports(latest=latest)
        if not df.empty:
            print(f"✅ Got {len(df)} AS price records from historical reports")
            return df
        
        # Approach 2: Try direct API (original method)
        df = fetch_ancillary_from_api(
            self.endpoints, self.schema, self.lookback_hours, self.timeout, latest=latest
        )
        if not df.empty:
            print(f"✅ Got {len(df)} AS price records from direct API")
            return df
        
        # Approach 3: Generate synthetic AS prices as fallback
        print("⚠️  No real AS prices available, will use synthetic prices")
        return pd.DataFrame()
    
    def _fetch_ancillary_from_reports(self, *, latest: bool = True) -> pd.DataFrame:
        """Try to fetch AS prices from historical reports."""
        try:
            # Use the working Excel report approach
            reports = self.fetch_historical_reports(
                report_type='ancillary_prices',
                use_cache=True
            )
            
            if not reports:
                return pd.DataFrame()
            
            # Download most recent report
            latest_report = reports[0]
            doc_id = latest_report.get('DocID')
            
            df = self.download_historical_report(
                doc_id=doc_id,
                report_name='ancillary_prices_latest',
                use_cache=True
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Process the data to extract AS prices
            # This appears to be LMP data, so we'll derive AS prices from it
            df = derive_as_prices_from_lmp(df)
            
            return df
            
        except Exception as e:
            print(f"Warning: Historical AS reports fetch failed: {e}")
            return pd.DataFrame()

    def fetch_historical_reports(
        self,
        report_type: str = "real_time_lmp",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """Fetch list of available historical reports from ERCOT archive.
        
        Args:
            report_type: Type of report (real_time_lmp, real_time_load, etc.)
            start_date: Filter reports published after this date
            end_date: Filter reports published before this date
            use_cache: Whether to use cached report list
        
        Returns:
            List of report metadata dictionaries
        """
        report_type_id = self.endpoints.report_type_ids.get(report_type)
        if not report_type_id:
            raise ValueError(f"Unknown report type: {report_type}. Available: {list(self.endpoints.report_type_ids.keys())}")
        
        # Check cache first
        cache_file = self.cache_dir / f"report_list_{report_type}_{datetime.now().date()}.json"
        if use_cache and cache_file.exists():
            with open(cache_file, 'r') as f:
                reports = json.load(f)
            print(f"📂 Loaded {len(reports)} reports from cache: {cache_file.name}")
        else:
            # Fetch from ERCOT API
            url = self.endpoints.ice_doc_list_base
            params = {"reportTypeId": report_type_id}
            
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                # Extract document list
                doc_list = data.get('ListDocsByRptTypeRes', {}).get('DocumentList', [])
                reports = [doc.get('Document', {}) for doc in doc_list]
                
                # Cache the results
                with open(cache_file, 'w') as f:
                    json.dump(reports, f, indent=2)
                print(f"✅ Fetched {len(reports)} reports from ERCOT API")
                
            except Exception as e:
                print(f"❌ Failed to fetch report list: {e}")
                return []
        
        # Filter by date range if provided
        if start_date or end_date:
            filtered = []
            for report in reports:
                pub_date_str = report.get('PublishDate', '')
                try:
                    pub_date = pd.to_datetime(pub_date_str)
                    if start_date and pub_date < start_date:
                        continue
                    if end_date and pub_date > end_date:
                        continue
                    filtered.append(report)
                except Exception:
                    continue
            reports = filtered
            print(f"🔍 Filtered to {len(reports)} reports in date range")
        
        return reports

    def download_historical_report(
        self,
        doc_id: str,
        report_name: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Download and parse a specific historical report.
        
        Args:
            doc_id: Document ID from ERCOT
            report_name: Friendly name of the report
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with report data, or None if download fails
        """
        # Check cache first (use pickle for now, parquet requires pyarrow)
        cache_file = self.cache_dir / f"{report_name}.pkl"
        if use_cache and cache_file.exists():
            df = pd.read_pickle(cache_file)
            print(f"📂 Loaded from cache: {cache_file.name} ({len(df)} rows)")
            return df
        
        # Download from ERCOT
        url = self.endpoints.ice_doc_download_base
        params = {"doclookupId": doc_id}
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Parse ZIP file
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
                excel_files = [name for name in zf.namelist() if name.endswith(('.xlsx', '.xls'))]
                
                if csv_files:
                    # Read first CSV file
                    with zf.open(csv_files[0]) as f:
                        df = pd.read_csv(f)
                elif excel_files:
                    # Read first Excel file
                    print(f"📊 Reading Excel file: {excel_files[0]}")
                    with zf.open(excel_files[0]) as f:
                        # Extract Excel content to temp location
                        excel_content = f.read()
                        temp_file = BytesIO(excel_content)
                        df = pd.read_excel(temp_file, engine='openpyxl')
                else:
                    print(f"⚠️  No CSV or Excel files in archive: {report_name}")
                    return None
                
                # Apply schema mapping if it's LMP data
                if 'LMP' in str(df.columns):
                    df = df.rename(columns=self.schema.lmp_schema)
                elif 'ERCOT' in str(df.columns):
                    df = df.rename(columns=self.schema.load_schema)
                
                # Cache the results (use pickle for now)
                df.to_pickle(cache_file)
                print(f"💾 Downloaded and cached: {report_name} ({len(df)} rows)")
                
                return df
                
        except Exception as e:
            print(f"❌ Failed to download report {report_name}: {e}")
            return None

    def fetch_historical_data(
        self,
        report_type: str = "real_time_lmp",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_reports: Optional[int] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch and aggregate historical data from multiple reports.
        
        Args:
            report_type: Type of report to fetch
            start_date: Start date for data range
            end_date: End date for data range
            max_reports: Maximum number of reports to download (None = all)
            use_cache: Whether to use cached data
        
        Returns:
            Aggregated DataFrame with all historical data
        """
        # Fetch report list
        reports = self.fetch_historical_reports(
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        if not reports:
            print("❌ No reports available")
            return pd.DataFrame()
        
        # Limit number of reports if specified
        if max_reports:
            reports = reports[:max_reports]
            print(f"📊 Processing {len(reports)} reports (limited by max_reports={max_reports})")
        else:
            print(f"📊 Processing all {len(reports)} available reports")
        
        # Download and aggregate
        dfs = []
        for i, report in enumerate(reports, 1):
            doc_id = report.get('DocID')
            friendly_name = report.get('FriendlyName', f'report_{doc_id}')
            
            print(f"[{i}/{len(reports)}] Processing {friendly_name}...")
            
            df = self.download_historical_report(
                doc_id=doc_id,
                report_name=friendly_name,
                use_cache=use_cache
            )
            
            if df is not None and not df.empty:
                dfs.append(df)
        
        if not dfs:
            print("❌ No data downloaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n✅ Combined {len(dfs)} reports into {len(combined)} total rows")
        
        # Sort by timestamp if available
        if 'timestamp' in combined.columns:
            combined = combined.sort_values('timestamp').reset_index(drop=True)
        elif 'delivery_date' in combined.columns:
            combined['timestamp'] = pd.to_datetime(combined['delivery_date'])
            if 'delivery_hour' in combined.columns:
                combined['timestamp'] += pd.to_timedelta((combined['delivery_hour'] - 1) * 60, unit='min')
            if 'delivery_interval' in combined.columns:
                combined['timestamp'] += pd.to_timedelta((combined['delivery_interval'] - 1) * 5, unit='min')
            combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        return combined

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """Clear cached historical data.
        
        Args:
            older_than_days: Only clear files older than this many days (None = clear all)
        
        Returns:
            Number of files deleted
        """
        count = 0
        cutoff = datetime.now() - timedelta(days=older_than_days) if older_than_days else None
        
        for file in self.cache_dir.glob('*'):
            if cutoff:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                if file_time > cutoff:
                    continue
            file.unlink()
            count += 1
        
        print(f"🗑️  Cleared {count} cached files from {self.cache_dir}")
        return count


# Export build_observation_operator for backward compatibility
__all__ = ["ERCOTDataClient", "build_observation_operator"]

