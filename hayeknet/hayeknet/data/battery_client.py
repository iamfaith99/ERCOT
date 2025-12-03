"""Enhanced ERCOT data client for battery trading analysis with historical data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from hayeknet.data.client import ERCOTDataClient


@dataclass
class BatteryDataClient(ERCOTDataClient):
    """Extended data client for battery trading analysis with historical lookback."""
    
    # Historical data parameters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_ancillary: bool = True
    
    def fetch_historical_rtc_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical ERCOT data for battery strategy analysis.
        
        Parameters
        ----------
        start_date : datetime, optional
            Start of historical period (default: 7 days ago)
        end_date : datetime, optional
            End of historical period (default: now)
            
        Returns
        -------
        pd.DataFrame
            Historical market data with columns:
            - timestamp: UTC timestamp
            - net_load_mw: System load
            - lmp_usd: Locational marginal price
            - reg_up_price: Regulation up price (if available)
            - reg_down_price: Regulation down price (if available)
            - rrs_price: Responsive reserve service price (if available)
            - ecrs_price: ERCOT contingency reserve service price (if available)
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.utcnow()
            
        print(f"📅 Fetching historical data: {start_date.date()} to {end_date.date()}")
        
        # Try multi-source archived data first
        df = self._fetch_archived_multi_source_data(start_date, end_date)
        
        # If archived data fails, try real-time data
        if df.empty:
            df = self._fetch_historical_dashboard_data(start_date, end_date)
        
        # If real data fails, generate synthetic
        if df.empty:
            print("⚠️  Real data unavailable, generating synthetic historical data")
            df = self._generate_historical_synthetic(start_date, end_date)
        
        # Add ancillary service prices if not already present
        if self.include_ancillary and 'reg_up_price' not in df.columns:
            df = self._add_ancillary_prices(df)
        
        # Add derived metrics for battery analysis
        df = self._add_battery_signals(df)
        
        return df.sort_values("timestamp").reset_index(drop=True)
    
    def _fetch_archived_multi_source_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch archived multi-source ERCOT data from the enhanced daily collector.
        
        Looks for data in the multi-source archive created by the enhanced
        daily data collector, which combines real-time LMP, DAM, system lambda,
        and ancillary service prices.
        """
        from pathlib import Path
        import glob
        
        # Path to multi-source archived data
        project_root = Path(__file__).resolve().parents[1]
        archive_base = project_root / "data" / "archive"
        
        dfs = []
        
        # Look for archived parquet files in date range
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Try multi-source combined archive first
            multi_source_pattern = archive_base / f"ercot_multi_source_{date_str}_*.parquet"
            multi_source_files = glob.glob(str(multi_source_pattern))
            
            if multi_source_files:
                # Use latest file for the date
                latest_file = max(multi_source_files)
                try:
                    df_day = pd.read_parquet(latest_file)
                    
                    # Ensure timestamp column exists and is datetime
                    if 'timestamp' in df_day.columns:
                        df_day['timestamp'] = pd.to_datetime(df_day['timestamp'], utc=True)
                    elif 'DeliveryDate' in df_day.columns:
                        # Handle ERCOT timestamp format
                        df_day['timestamp'] = pd.to_datetime(df_day['DeliveryDate'], utc=True)
                        if 'DeliveryHour' in df_day.columns and 'DeliveryInterval' in df_day.columns:
                            df_day['timestamp'] += pd.to_timedelta(
                                (df_day['DeliveryHour'] - 1) * 60 + (df_day['DeliveryInterval'] - 1) * 5,
                                unit='min'
                            )
                    
                    # Standardize column names
                    if 'LMP' in df_day.columns and 'lmp_usd' not in df_day.columns:
                        df_day['lmp_usd'] = df_day['LMP']
                    
                    dfs.append(df_day)
                    print(f"📂 Loaded multi-source data for {date_str}: {len(df_day)} records")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load multi-source data for {date_str}: {e}")
            
            else:
                # Fallback to individual source files
                lmp_pattern = archive_base / f"ercot_lmp_{date_str}_*.parquet"
                lmp_files = glob.glob(str(lmp_pattern))
                
                if lmp_files:
                    try:
                        latest_lmp = max(lmp_files)
                        df_lmp = pd.read_parquet(latest_lmp)
                        
                        # Basic timestamp handling for LMP data
                        if 'timestamp' not in df_lmp.columns and 'DeliveryDate' in df_lmp.columns:
                            df_lmp['timestamp'] = pd.to_datetime(df_lmp['DeliveryDate'], utc=True)
                            if 'DeliveryHour' in df_lmp.columns:
                                df_lmp['timestamp'] += pd.to_timedelta((df_lmp['DeliveryHour'] - 1) * 60, unit='min')
                            if 'DeliveryInterval' in df_lmp.columns:
                                df_lmp['timestamp'] += pd.to_timedelta((df_lmp['DeliveryInterval'] - 1) * 5, unit='min')
                        
                        if 'LMP' in df_lmp.columns:
                            df_lmp['lmp_usd'] = df_lmp['LMP']
                        
                        # Add estimated load if not present
                        if 'net_load_mw' not in df_lmp.columns and 'lmp_usd' in df_lmp.columns:
                            # Reverse-estimate load from LMP using simple model
                            df_lmp['net_load_mw'] = 45000 + (df_lmp['lmp_usd'] - 25) / 0.001
                        
                        dfs.append(df_lmp)
                        print(f"📂 Loaded LMP archive for {date_str}: {len(df_lmp)} records")
                        
                    except Exception as e:
                        print(f"⚠️  Failed to load LMP archive for {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        if not dfs:
            print(f"📭 No archived data found for {start_date.date()} to {end_date.date()}")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined = pd.concat(dfs, ignore_index=True)
        
        # Filter to requested time range
        if 'timestamp' in combined.columns:
            combined = combined[
                (combined['timestamp'] >= start_date) & 
                (combined['timestamp'] <= end_date)
            ]
        
        print(f"✅ Loaded {len(combined)} archived records from {len(dfs)} files")
        return combined
    
    def _fetch_historical_dashboard_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Attempt to fetch historical data from ERCOT dashboard APIs."""
        try:
            # For now, dashboard only provides current data
            # Future: implement historical API endpoint when available
            print("ℹ️  Dashboard API provides current data only, using synthetic for historical")
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️  Historical fetch failed: {e}")
            return pd.DataFrame()
    
    def _generate_historical_synthetic(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic historical ERCOT data.
        
        Includes:
        - Seasonal patterns (summer peak demand)
        - Daily cycles (morning/evening ramps)
        - Weekend effects
        - Price spikes during high load
        - Correlation between load and LMP
        """
        # Generate 5-minute timestamps
        timestamps = pd.date_range(start_date, end_date, freq="5min")
        n = len(timestamps)
        
        # Extract temporal features
        hours = np.array(timestamps.hour + timestamps.minute / 60)
        day_of_week = np.array(timestamps.dayofweek)
        month = np.array(timestamps.month)
        
        # Seasonal component (summer peak)
        seasonal = np.where(
            (month >= 6) & (month <= 9),  # Summer
            1.15,  # 15% higher load
            np.where(
                (month == 12) | (month <= 2),  # Winter
                1.05,  # 5% higher load
                1.0,  # Shoulder months
            )
        )
        
        # Daily cycle (morning and evening peaks)
        daily_pattern = (
            0.7  # Baseline
            + 0.15 * np.sin(2 * np.pi * (hours - 6) / 24)  # Primary cycle
            + 0.10 * np.sin(4 * np.pi * (hours - 8) / 24)  # Morning peak
            + 0.05 * np.sin(4 * np.pi * (hours - 18) / 24)  # Evening peak
        )
        
        # Weekend effect (lower load)
        weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)
        
        # Base load with all patterns
        base_load = 55_000  # MW baseline
        net_load = (
            base_load * seasonal * daily_pattern * weekend_factor
            + np.random.normal(0, 2000, n)  # Noise
        )
        
        # LMP with load correlation and volatility
        # Higher load = higher prices, with occasional spikes
        load_normalized = (net_load - net_load.mean()) / net_load.std()
        base_lmp = 35 + 15 * load_normalized  # Load-price correlation
        
        # Add price spikes during extreme load (> 2 std above mean)
        spike_prob = 1 / (1 + np.exp(-3 * (load_normalized - 2)))  # Sigmoid
        price_spikes = np.random.binomial(1, spike_prob) * np.random.gamma(2, 50, n)
        
        lmp = np.maximum(base_lmp + price_spikes + np.random.normal(0, 5, n), 0)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "net_load_mw": net_load,
            "lmp_usd": lmp,
        })
    
    def _add_ancillary_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ancillary service prices with realistic patterns.
        
        Uses real ancillary price data if available from multi-source archives,
        otherwise generates synthetic prices.
        
        Ancillary prices typically:
        - Reg Up/Down: $5-30/MW
        - RRS (spinning reserve): $10-50/MW
        - ECRS (non-spinning): $5-30/MW
        - Higher during tight supply conditions
        """
        # Check if we already have real ancillary price data
        has_real_as_data = any(col in df.columns for col in 
                              ['reg_up_price', 'reg_down_price', 'rrs_price', 'ecrs_price'])
        
        if has_real_as_data:
            print("✅ Using real ancillary service prices from archived data")
            # Fill any missing AS price columns with reasonable defaults
            if 'reg_up_price' not in df.columns:
                df['reg_up_price'] = 15.0
            if 'reg_down_price' not in df.columns:
                df['reg_down_price'] = 8.0
            if 'rrs_price' not in df.columns:
                df['rrs_price'] = 20.0
            if 'ecrs_price' not in df.columns:
                df['ecrs_price'] = 12.0
            return df
        
        print("🎲 Generating synthetic ancillary service prices")
        
        n = len(df)
        
        # Scarcity indicator (high load periods)
        if "net_load_mw" in df.columns:
            load_percentile = df["net_load_mw"].rank(pct=True)
            scarcity = np.maximum(0, (load_percentile - 0.7) * 3.33)  # 0-1 scale
        else:
            scarcity = np.zeros(n)
        
        # Regulation Up (AGC control)
        df["reg_up_price"] = (
            8.0  # Base price
            + 15.0 * scarcity  # Scarcity premium
            + np.random.gamma(2, 3, n)  # Variability
        )
        
        # Regulation Down (typically cheaper)
        df["reg_down_price"] = (
            5.0
            + 8.0 * scarcity
            + np.random.gamma(2, 2, n)
        )
        
        # Responsive Reserve Service (RRS) - spinning reserve
        df["rrs_price"] = (
            12.0
            + 25.0 * scarcity
            + np.random.gamma(2, 4, n)
        )
        
        # ERCOT Contingency Reserve Service (ECRS) - non-spinning
        df["ecrs_price"] = (
            7.0
            + 15.0 * scarcity
            + np.random.gamma(2, 3, n)
        )
        
        return df
    
    def _add_battery_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived signals useful for battery trading strategies.
        
        Signals:
        - price_momentum: Rate of price change
        - load_forecast_error: Proxy for uncertainty
        - arbitrage_opportunity: Price spread for charge/discharge
        - ancillary_premium: AS price / energy price ratio
        """
        # Price momentum (5-period moving average of changes)
        df["price_momentum"] = (
            df["lmp_usd"]
            .diff()
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )
        
        # Load forecast error proxy (volatility)
        df["load_volatility"] = (
            df["net_load_mw"]
            .rolling(window=12, min_periods=1)  # 1 hour
            .std()
            .fillna(0)
        )
        
        # Arbitrage opportunity (future price - current price)
        # Use 12-period (1 hour) ahead as proxy
        df["price_forecast"] = (
            df["lmp_usd"]
            .rolling(window=12, min_periods=1)
            .mean()
            .shift(-12)
            .fillna(df["lmp_usd"])
        )
        df["arbitrage_signal"] = df["price_forecast"] - df["lmp_usd"]
        
        # Ancillary service premium (if AS prices available)
        if "reg_up_price" in df.columns:
            df["ancillary_premium"] = (
                df["reg_up_price"] + df["rrs_price"]
            ) / (df["lmp_usd"] + 1e-6)  # Avoid division by zero
        
        return df
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets chronologically.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset
        train_frac : float
            Fraction for training (0.8 = 80% train, 20% test)
            
        Returns
        -------
        train_df, test_df : tuple of DataFrames
        """
        n = len(df)
        split_idx = int(n * train_frac)
        
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        
        print(f"📊 Data split: {len(train_df)} train, {len(test_df)} test")
        print(f"   Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"   Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        
        return train_df, test_df
    
    def compute_market_statistics(self, df: pd.DataFrame) -> dict:
        """
        Compute key market statistics for battery profitability analysis.
        
        Returns
        -------
        dict
            Market statistics including:
            - Price statistics (mean, volatility, spikes)
            - Arbitrage opportunities
            - Ancillary service economics
        """
        stats = {
            # Energy market
            "mean_lmp": float(df["lmp_usd"].mean()),
            "std_lmp": float(df["lmp_usd"].std()),
            "max_lmp": float(df["lmp_usd"].max()),
            "price_volatility_hourly": float(
                df["lmp_usd"].rolling(12).std().mean()
            ),
            
            # Price spikes (>$100/MWh)
            "spike_frequency": float((df["lmp_usd"] > 100).mean()),
            "mean_spike_price": float(df.loc[df["lmp_usd"] > 100, "lmp_usd"].mean())
            if (df["lmp_usd"] > 100).any() else 0.0,
            
            # Arbitrage potential
            "mean_price_spread_1h": float(
                (df["lmp_usd"].rolling(12).max() - df["lmp_usd"].rolling(12).min()).mean()
            ),
            "arbitrage_opportunities_per_day": float(
                (df["arbitrage_signal"].abs() > 5).sum() / 7.0  # Approximate: total / 7 days
            ) if "arbitrage_signal" in df.columns else 0.0,
        }
        
        # Ancillary service statistics
        if "reg_up_price" in df.columns:
            stats.update({
                "mean_reg_up": float(df["reg_up_price"].mean()),
                "mean_reg_down": float(df["reg_down_price"].mean()),
                "mean_rrs": float(df["rrs_price"].mean()),
                "mean_ecrs": float(df["ecrs_price"].mean()),
                "ancillary_premium_ratio": float(df["ancillary_premium"].mean())
                if "ancillary_premium" in df.columns else 0.0,
            })
        
        return stats
