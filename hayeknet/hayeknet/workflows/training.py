"""Agent training decision logic."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def should_train_agents(data_dir: Path, marl_state_file: Path) -> tuple[bool, str]:
    """Determine if agent training is needed based on market conditions.
    
    Returns:
        (should_train, reason): Boolean decision and explanation string
    """
    # Check days since last training
    if marl_state_file.exists():
        try:
            with open(marl_state_file, 'r') as f:
                state = json.load(f)
                last_training = state.get('last_training_date', '2020-01-01T00:00:00')
                last_date = datetime.fromisoformat(last_training)
                days_since = (datetime.now() - last_date).days
                
                # Always train if it's been more than 7 days
                if days_since >= 7:
                    return True, f"Weekly training refresh ({days_since} days since last training)"
                
                print(f"   📅 Last trained {days_since} days ago")
        except Exception as e:
            print(f"   ⚠️  Error reading training state: {e}")
            return True, "Training needed: Error reading state file"
    else:
        return True, "Training needed: No previous training state found"
    
    # Check market volatility from recent parquet files
    if data_dir.exists():
        try:
            parquet_files = sorted(data_dir.glob('ercot_lmp_*.parquet'))
            if parquet_files:
                # Look at last 3 days of data
                recent_files = parquet_files[-3:]
                volatilities = []
                
                for pq_file in recent_files:
                    try:
                        df = pd.read_parquet(pq_file)
                        # Check for lmp_usd column (standardized)
                        if 'lmp_usd' in df.columns:
                            vol = df['lmp_usd'].std()
                            volatilities.append(vol)
                    except Exception:
                        continue
                
                if volatilities:
                    avg_volatility = np.mean(volatilities)
                    print(f"   📊 Recent market volatility: ${avg_volatility:.2f}/MWh")
                    
                    # Train if volatility is high
                    if avg_volatility > 15.0:
                        return True, f"High market volatility detected (${avg_volatility:.2f} > $15.00/MWh)"
        except Exception as e:
            print(f"   ⚠️  Error checking volatility: {e}")
    
    # Market is stable, skip training
    return False, "Market conditions stable, training skipped to save compute time"

