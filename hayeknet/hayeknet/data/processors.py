"""Data processing and transformation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_observation_operator(obs_matrix: np.ndarray) -> np.ndarray:
    """Return an identity observation operator sized to the observation matrix."""
    obs_dim = obs_matrix.shape[0]
    return np.eye(obs_dim, dtype=float)


def derive_as_prices_from_lmp(lmp_df: pd.DataFrame) -> pd.DataFrame:
    """Derive realistic AS prices from LMP data patterns."""
    if lmp_df.empty:
        return pd.DataFrame()
    
    try:
        # Extract timestamp info
        if 'Delivery Date' in lmp_df.columns:
            lmp_df['timestamp'] = pd.to_datetime(lmp_df['Delivery Date'])
            if 'Hour Ending' in lmp_df.columns:
                # Convert hour ending to proper timestamp
                hour_ending = lmp_df['Hour Ending'].str.replace(':00', '').astype(int) - 1
                lmp_df['timestamp'] += pd.to_timedelta(hour_ending, unit='h')
        
        # Get system average LMP
        if 'Settlement Point Price' in lmp_df.columns:
            # Filter for hub average or system-wide prices
            hub_data = lmp_df[lmp_df['Settlement Point'].str.contains('HUBAVG|HB_BUSAVG', na=False)]
            
            if hub_data.empty:
                # Use all data and average
                system_lmp = lmp_df.groupby('timestamp')['Settlement Point Price'].mean().reset_index()
            else:
                system_lmp = hub_data.groupby('timestamp')['Settlement Point Price'].mean().reset_index()
            
            system_lmp.rename(columns={'Settlement Point Price': 'lmp_usd'}, inplace=True)
            
            # Derive AS prices from LMP using realistic relationships
            as_records = []
            
            for _, row in system_lmp.iterrows():
                timestamp = row['timestamp']
                lmp = row['lmp_usd']
                
                # Realistic AS price derivation based on ERCOT patterns
                reg_up = max(5.0, lmp * 0.3 + np.random.gamma(2, 3))  # RegUp typically 30% of LMP + premium
                reg_down = max(2.0, lmp * 0.15 + np.random.gamma(2, 2))  # RegDown typically 15% of LMP
                rrs = max(8.0, lmp * 0.4 + np.random.gamma(2, 4))  # RRS typically 40% of LMP + premium
                ecrs = max(4.0, lmp * 0.25 + np.random.gamma(2, 3))  # ECRS typically 25% of LMP
                
                as_records.extend([
                    {'timestamp': timestamp, 'ancillary_type': 'REGUP', 'market_clearing_price': reg_up},
                    {'timestamp': timestamp, 'ancillary_type': 'REGDN', 'market_clearing_price': reg_down}, 
                    {'timestamp': timestamp, 'ancillary_type': 'RRS', 'market_clearing_price': rrs},
                    {'timestamp': timestamp, 'ancillary_type': 'ECRS', 'market_clearing_price': ecrs},
                ])
            
            return pd.DataFrame(as_records)
        
    except Exception as e:
        print(f"Warning: AS price derivation failed: {e}")
        
    return pd.DataFrame()

