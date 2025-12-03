#!/usr/bin/env python3
"""Retrain all ML models with updated RTC+B features.

This script:
1. Retrains all 3 MARL agents (Battery, Solar, Wind) with RTC+B environment
2. Retrains single-agent RL model with updated RTCEnv
3. Verifies Bayesian reasoner compatibility
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from hayeknet.core.agents import ResourceType
from hayeknet.ml.qse_agents import MARLSystem
from hayeknet.ml.rl import RLTrainer
from hayeknet.ml.bayesian import BayesianReasoner
from hayeknet.data.client import ERCOTDataClient
from hayeknet.core.market import MarketDesign

# Import RTCEnv - handle both possible paths
try:
    from envs.rtc_env import RTCEnv, RTCEnvConfig
except ImportError:
    import sys
    sys.path.insert(0, str(project_root))
    from envs.rtc_env import RTCEnv, RTCEnvConfig


def retrain_marl_agents(force: bool = True) -> dict:
    """Retrain all MARL agents with RTC+B features."""
    print("="*60)
    print("RETRAINING MARL AGENTS")
    print("="*60)
    
    base_dir = project_root
    data_dir = base_dir / 'data' / 'archive' / 'ercot_lmp'
    model_dir = base_dir / 'models' / 'qse_agents'
    marl_state_file = base_dir / 'models' / 'marl_state.json'
    
    # Check for data
    parquet_files = list(data_dir.glob('ercot_lmp_*.parquet'))
    if not parquet_files:
        print(f"⚠️  No training data found in {data_dir}")
        return {'status': 'failed', 'reason': 'no_data'}
    
    print(f"📊 Found {len(parquet_files)} data files")
    
    # Initialize MARL system
    marl_system = MARLSystem(model_dir=model_dir)
    
    # Clear existing models if force retrain
    if force:
        print("🔄 Force retraining: clearing existing models...")
        for resource_type in [ResourceType.BATTERY, ResourceType.SOLAR, ResourceType.WIND]:
            model_path = model_dir / f"{resource_type.value}_latest.zip"
            if model_path.exists():
                model_path.unlink()
                print(f"   Deleted: {model_path.name}")
    
    # Load existing state if available
    if marl_state_file.exists() and not force:
        marl_system.load_state(marl_state_file)
        print("   📂 Loaded existing MARL state")
    else:
        print("   🆕 Initializing new MARL agents")
    
    # Train agents
    print("\n🤖 Training agents on historical data...")
    training_start = time.time()
    
    try:
        training_metrics = marl_system.train_incremental(
            data_dir=data_dir,
            timesteps_per_day=10000,  # Full training
            save_checkpoints=True
        )
        
        training_time = time.time() - training_start
        print(f"\n✅ Training complete!")
        print(f"   Time: {training_time:.1f}s")
        print(f"   Agents trained: {len(training_metrics)}")
        
        for resource_type, metrics in training_metrics.items():
            print(f"   - {resource_type}: {metrics['timesteps']} timesteps, {metrics['total_data_points']} data points")
        
        # Save state
        marl_system.save_state(marl_state_file)
        print(f"   💾 State saved: {marl_state_file}")
        
        return {
            'status': 'success',
            'training_time': training_time,
            'metrics': training_metrics
        }
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


def retrain_single_agent_rl(force: bool = True) -> dict:
    """Retrain single-agent RL model with updated RTCEnv."""
    print("\n" + "="*60)
    print("RETRAINING SINGLE-AGENT RL")
    print("="*60)
    
    # Load market data
    print("📊 Loading market data...")
    client = ERCOTDataClient()
    data_dir = project_root / 'data' / 'archive' / 'ercot_lmp'
    parquet_files = sorted(data_dir.glob('ercot_lmp_*.parquet'))
    
    if not parquet_files:
        print("⚠️  No training data found")
        return {'status': 'failed', 'reason': 'no_data'}
    
    # Load most recent data file
    df = pd.read_parquet(parquet_files[-1])
    
    # Standardize column names
    if 'lmp_usd' not in df.columns and 'LMP' in df.columns:
        df['lmp_usd'] = df['LMP']
    
    print(f"✅ Loaded {len(df):,} observations")
    
    # Create beliefs (simplified - use mean LMP as belief)
    beliefs = np.full(len(df), 0.5, dtype=float)
    
    # Create environment with RTC+B
    config = RTCEnvConfig(
        max_bid_mw=100.0,
        market_design=MarketDesign.RTC_PLUS_B,
    )
    
    print("🔧 Creating RTC+B environment...")
    env_factory = lambda: RTCEnv(df, beliefs, config)
    
    # Train model
    print("🤖 Training RL agent...")
    training_start = time.time()
    
    try:
        trainer = RLTrainer(
            vector_envs=4,
            total_timesteps=10000,  # Full training
        )
        model, rl_info = trainer.train(env_factory)
        
        training_time = time.time() - training_start
        print(f"\n✅ Training complete!")
        print(f"   Time: {training_time:.1f}s")
        print(f"   Timesteps: {rl_info.get('timesteps', 'unknown')}")
        
        # Save model
        model_dir = project_root / 'models' / 'rl'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'rtc_env_latest.zip'
        model.save(model_path)
        print(f"   💾 Model saved: {model_path}")
        
        return {
            'status': 'success',
            'training_time': training_time,
            'model_path': str(model_path),
        }
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


def verify_bayesian_reasoner() -> dict:
    """Verify Bayesian reasoner works with new data structure."""
    print("\n" + "="*60)
    print("VERIFYING BAYESIAN REASONER")
    print("="*60)
    
    try:
        reasoner = BayesianReasoner(prior_high=0.35)
        
        # Test with sample data
        test_values = [25.0, 30.0, 50.0, 100.0]
        beliefs = []
        
        for val in test_values:
            belief = reasoner.update(val)
            beliefs.append(belief)
            print(f"   LMP=${val:.2f} → Belief={belief:.3f}")
        
        print("\n✅ Bayesian reasoner verified")
        return {
            'status': 'success',
            'test_beliefs': beliefs
        }
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def main():
    """Main retraining function."""
    print("="*60)
    print("MODEL RETRAINING WITH RTC+B FEATURES")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description="Retrain ML models with RTC+B features")
    parser.add_argument('--marl-only', action='store_true', help='Only retrain MARL agents')
    parser.add_argument('--rl-only', action='store_true', help='Only retrain single-agent RL')
    parser.add_argument('--verify-only', action='store_true', help='Only verify Bayesian reasoner')
    parser.add_argument('--force', action='store_true', default=True, help='Force retrain (clear existing models)')
    
    args = parser.parse_args()
    
    results = {}
    
    # Retrain MARL agents
    if not args.rl_only and not args.verify_only:
        results['marl'] = retrain_marl_agents(force=args.force)
    
    # Retrain single-agent RL
    if not args.marl_only and not args.verify_only:
        results['rl'] = retrain_single_agent_rl(force=args.force)
    
    # Verify Bayesian reasoner
    if not args.marl_only and not args.rl_only:
        results['bayesian'] = verify_bayesian_reasoner()
    
    # Summary
    print("\n" + "="*60)
    print("RETRAINING SUMMARY")
    print("="*60)
    
    for component, result in results.items():
        status = result.get('status', 'unknown')
        if status == 'success':
            print(f"✅ {component.upper()}: Success")
        else:
            print(f"❌ {component.upper()}: Failed - {result.get('error', result.get('reason', 'unknown'))}")
    
    print("\n✅ Retraining complete!")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main()

