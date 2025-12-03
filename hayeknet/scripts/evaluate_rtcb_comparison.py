#!/usr/bin/env python3
"""Comprehensive evaluation script comparing SCED vs RTC+B market designs.

This script runs parallel simulations under different market designs:
1. Baseline SCED (current market)
2. RTC+B (full implementation)
3. RTC+B (no ASDCs)
4. RTC+B (no SOC checks)

Generates comparative metrics and visualizations for research paper.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from hayeknet.core.battery import BatterySimulator, BatterySpecs
from hayeknet.core.market import (
    MarketDesign,
    SCEDSimulator,
    RTCPlusBSimulator,
    compare_market_designs,
)
from hayeknet.data.client import ERCOTDataClient
from hayeknet.strategies.arbitrage import SimpleArbitrageStrategy


def load_market_data(data_dir: Path, days: int = 7) -> pd.DataFrame:
    """Load market data for evaluation."""
    client = ERCOTDataClient()
    
    # Try to load from archive first
    archive_dir = data_dir / "archive" / "ercot_lmp"
    parquet_files = sorted(archive_dir.glob("ercot_lmp_*.parquet"))
    
    if parquet_files:
        # Load most recent files
        dfs = []
        for pq_file in parquet_files[-days:]:
            df = pd.read_parquet(pq_file)
            dfs.append(df)
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            
            # Standardize column names
            if 'lmp_usd' not in combined.columns and 'LMP' in combined.columns:
                combined['lmp_usd'] = combined['LMP']
            
            # Add AS prices if missing
            if 'reg_up_price' not in combined.columns:
                combined['reg_up_price'] = combined['lmp_usd'] * 0.5
                combined['reg_down_price'] = combined['lmp_usd'] * 0.3
                combined['rrs_price'] = combined['lmp_usd'] * 0.4
                combined['ecrs_price'] = combined['lmp_usd'] * 0.35
            
            return combined
    
    # Fallback: fetch recent data
    print("⚠️  No archived data found, fetching recent data...")
    df, _ = client.fetch_rtc_like()
    return df


def run_simulation_variant(
    market_data: pd.DataFrame,
    battery_specs: BatterySpecs,
    market_design: MarketDesign,
    variant_name: str,
    asdc_enabled: bool = True,
    soc_checks_enabled: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run simulation under specific market design variant.
    
    Parameters
    ----------
    market_data : pd.DataFrame
        Market data for simulation
    battery_specs : BatterySpecs
        Battery specifications
    market_design : MarketDesign
        Market design (SCED or RTC_PLUS_B)
    variant_name : str
        Name of this variant for reporting
    asdc_enabled : bool
        Whether ASDCs are enabled (RTC+B only)
    soc_checks_enabled : bool
        Whether SOC duration checks are enabled (RTC+B only)
    
    Returns
    -------
    results_df : pd.DataFrame
        Detailed results per interval
    summary : dict
        Summary statistics
    """
    print(f"\n{'='*60}")
    print(f"Running: {variant_name}")
    print(f"{'='*60}")
    
    # Initialize battery and simulator (reset for each variant)
    battery = BatterySimulator(battery_specs)
    battery.reset()  # Ensure clean state
    
    if market_design == MarketDesign.SCED:
        simulator = SCEDSimulator(battery)
    else:
        simulator = RTCPlusBSimulator(battery, asdc_enabled=asdc_enabled)
        # Note: SOC checks are always enabled in RTCPlusBSimulator
        # This parameter is for future use if we add a variant
    
    # Initialize strategy (fresh instance for each variant)
    strategy = SimpleArbitrageStrategy()
    
    # Run simulation
    results = []
    total_revenue = 0.0
    total_energy_revenue = 0.0
    total_as_revenue = 0.0
    
    for idx, row in market_data.iterrows():
        # Create market data series
        market_series = pd.Series({
            'lmp_usd': row.get('lmp_usd', row.get('LMP', 25.0)),
            'reg_up_price': row.get('reg_up_price', 15.0),
            'reg_down_price': row.get('reg_down_price', 8.0),
            'rrs_price': row.get('rrs_price', 20.0),
            'ecrs_price': row.get('ecrs_price', 12.0),
            'net_load_mw': row.get('net_load_mw', 60000.0),
            'timestamp': row.get('timestamp', idx),
        })
        
        # Generate bid
        bid = strategy.generate_bid(battery, market_series, market_design)
        
        # Clear market
        outcome = simulator.clear_market(bid, market_series)
        
        # Execute battery operation
        battery.step(
            outcome.energy_cleared_mw,
            reg_up_mw=outcome.reg_up_cleared_mw,
            reg_down_mw=outcome.reg_down_cleared_mw,
            rrs_mw=outcome.rrs_cleared_mw,
            ecrs_mw=outcome.ecrs_cleared_mw,
        )
        
        # Record results
        interval_revenue = outcome.total_revenue
        total_revenue += interval_revenue
        total_energy_revenue += outcome.energy_revenue
        total_as_revenue += (
            outcome.reg_up_revenue +
            outcome.reg_down_revenue +
            outcome.rrs_revenue +
            outcome.ecrs_revenue
        )
        
        results.append({
            'timestamp': market_series.get('timestamp', idx),
            'variant': variant_name,
            'lmp': market_series['lmp_usd'],
            'soc': battery.state.soc,
            'energy_cleared_mw': outcome.energy_cleared_mw,
            'reg_up_cleared_mw': outcome.reg_up_cleared_mw,
            'reg_down_cleared_mw': outcome.reg_down_cleared_mw,
            'rrs_cleared_mw': outcome.rrs_cleared_mw,
            'ecrs_cleared_mw': outcome.ecrs_cleared_mw,
            'energy_revenue': outcome.energy_revenue,
            'as_revenue': (
                outcome.reg_up_revenue +
                outcome.reg_down_revenue +
                outcome.rrs_revenue +
                outcome.ecrs_revenue
            ),
            'total_revenue': interval_revenue,
            'cumulative_revenue': total_revenue,
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    returns = results_df['total_revenue'].diff().fillna(0.0)
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(288) if len(returns) > 1 else 0.0
    
    summary = {
        'variant': variant_name,
        'market_design': market_design.value,
        'total_revenue': total_revenue,
        'energy_revenue': total_energy_revenue,
        'as_revenue': total_as_revenue,
        'as_revenue_pct': (total_as_revenue / total_revenue * 100) if total_revenue > 0 else 0.0,
        'total_cycles': battery.state.total_cycles,
        'cycles_per_day': battery.state.total_cycles / (len(results_df) / 288.0) if len(results_df) > 0 else 0.0,  # 288 intervals per day
        'mean_soc': results_df['soc'].mean(),
        'soc_std': results_df['soc'].std(),
        'soc_utilization': (results_df['soc'].max() - results_df['soc'].min()),
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': _calculate_max_drawdown(results_df['cumulative_revenue'].values),
        'num_intervals': len(results_df),
        'as_participation_pct': (
            (results_df[['reg_up_cleared_mw', 'reg_down_cleared_mw', 'rrs_cleared_mw', 'ecrs_cleared_mw']].sum(axis=1) > 0).sum() / len(results_df) * 100
        ),
    }
    
    print(f"✅ {variant_name} complete")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   AS Revenue: ${total_as_revenue:,.2f} ({summary['as_revenue_pct']:.1f}%)")
    print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"   AS Participation: {summary['as_participation_pct']:.1f}%")
    
    return results_df, summary


def _calculate_max_drawdown(portfolio_series: np.ndarray) -> float:
    """Calculate maximum drawdown from portfolio series."""
    if len(portfolio_series) == 0:
        return 0.0
    cummax = np.maximum.accumulate(portfolio_series)
    drawdown = (portfolio_series - cummax) / (cummax + 1e-8)
    return float(np.min(drawdown)) * 100


def generate_comparative_analysis(
    all_results: List[pd.DataFrame],
    all_summaries: List[Dict],
    output_dir: Path,
) -> None:
    """Generate comparative analysis and visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    
    # Save detailed results
    results_file = output_dir / "rtcb_comparison_results.csv"
    combined_df.to_csv(results_file, index=False)
    print(f"\n✅ Saved detailed results: {results_file}")
    
    # Save summary
    summary_file = output_dir / "rtcb_comparison_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Saved summary: {summary_file}")
    
    # Generate comparison table
    print("\n" + "="*60)
    print("COMPARATIVE RESULTS SUMMARY")
    print("="*60)
    print(summary_df[['variant', 'total_revenue', 'as_revenue_pct', 'sharpe_ratio', 'as_participation_pct']].to_string(index=False))
    
    # Calculate improvements
    if len(summary_df) > 1:
        baseline = summary_df[summary_df['variant'] == 'SCED (Baseline)']
        rtcb = summary_df[summary_df['variant'] == 'RTC+B (Full)']
        
        if not baseline.empty and not rtcb.empty:
            revenue_improvement = rtcb.iloc[0]['total_revenue'] - baseline.iloc[0]['total_revenue']
            revenue_improvement_pct = (revenue_improvement / baseline.iloc[0]['total_revenue'] * 100) if baseline.iloc[0]['total_revenue'] > 0 else 0.0
            
            print(f"\n📊 RTC+B vs SCED Improvement:")
            print(f"   Revenue: ${revenue_improvement:,.2f} ({revenue_improvement_pct:+.1f}%)")
            print(f"   AS Revenue %: {rtcb.iloc[0]['as_revenue_pct']:.1f}% vs {baseline.iloc[0]['as_revenue_pct']:.1f}%")
            print(f"   Sharpe Ratio: {rtcb.iloc[0]['sharpe_ratio']:.3f} vs {baseline.iloc[0]['sharpe_ratio']:.3f}")


def main():
    """Main evaluation function."""
    print("="*60)
    print("RTC+B COMPARATIVE EVALUATION")
    print("="*60)
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    output_dir = project_root / "research" / "rtcb_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load market data
    print("\n📊 Loading market data...")
    market_data = load_market_data(data_dir, days=7)
    print(f"✅ Loaded {len(market_data):,} observations")
    
    # Battery specifications (100 MW / 400 MWh)
    battery_specs = BatterySpecs(
        max_charge_mw=100.0,
        max_discharge_mw=100.0,
        capacity_mwh=400.0,
        min_soc=0.1,
        max_soc=0.9,
        initial_soc=0.5,
    )
    
    # Run all simulation variants
    all_results = []
    all_summaries = []
    
    # 1. Baseline SCED
    results, summary = run_simulation_variant(
        market_data,
        battery_specs,
        MarketDesign.SCED,
        "SCED (Baseline)",
    )
    all_results.append(results)
    all_summaries.append(summary)
    
    # 2. RTC+B (Full)
    results, summary = run_simulation_variant(
        market_data,
        battery_specs,
        MarketDesign.RTC_PLUS_B,
        "RTC+B (Full)",
        asdc_enabled=True,
        soc_checks_enabled=True,
    )
    all_results.append(results)
    all_summaries.append(summary)
    
    # 3. RTC+B (no ASDCs)
    results, summary = run_simulation_variant(
        market_data,
        battery_specs,
        MarketDesign.RTC_PLUS_B,
        "RTC+B (no ASDCs)",
        asdc_enabled=False,
        soc_checks_enabled=True,
    )
    all_results.append(results)
    all_summaries.append(summary)
    
    # Generate comparative analysis
    generate_comparative_analysis(all_results, all_summaries, output_dir)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

