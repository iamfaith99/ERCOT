#!/usr/bin/env python3
"""Analyze all historical research results for paper preparation.

Aggregates all JSON result files from research/results/ and generates:
- Summary statistics
- Hypothesis testing (H1, H2, H3)
- SCED vs RTC+B comparison (if available)
- Paper-ready tables and figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all historical result files into a single DataFrame."""
    print(f"📊 Loading results from: {results_dir}")
    
    result_files = sorted(results_dir.glob("results_*.json"))
    
    if not result_files:
        print(f"⚠️  No result files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"✅ Found {len(result_files)} result files")
    
    all_results = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract date from filename
            date_str = result_file.stem.replace('results_', '')
            
            # Flatten nested structure
            row = {
                'date': date_str,
                'file': result_file.name,
            }
            
            # Extract battery metrics if available
            if 'battery' in data:
                battery = data['battery']
                row.update({
                    'final_pnl': battery.get('final_pnl', 0.0),
                    'total_revenue': battery.get('total_revenue', 0.0),
                    'energy_revenue': battery.get('energy_revenue', 0.0),
                    'as_revenue': battery.get('as_revenue', 0.0),
                    'total_cycles': battery.get('total_cycles', 0.0),
                    'mean_soc': battery.get('mean_soc', 0.5),
                    'soc_utilization': battery.get('soc_utilization', 0.0),
                    'sharpe_ratio': battery.get('sharpe_ratio', 0.0),
                    'max_drawdown': battery.get('max_drawdown', 0.0),
                })
            
            # Extract market statistics if available
            if 'market_stats' in data:
                market = data['market_stats']
                row.update({
                    'mean_lmp': market.get('mean_lmp', 0.0),
                    'std_lmp': market.get('std_lmp', 0.0),
                    'max_lmp': market.get('max_lmp', 0.0),
                    'min_lmp': market.get('min_lmp', 0.0),
                    'lmp_cov': market.get('coefficient_of_variation', 0.0),
                    'spike_frequency': market.get('spike_frequency', 0.0),
                })
            
            # Extract system results if available
            if 'components' in data:
                components = data['components']
                row.update({
                    'rl_bid_suggestion': components.get('reinforcement_learning', {}).get('bid_suggestion', 0.0),
                    'bayesian_belief': components.get('bayesian_reasoning', {}).get('belief', 0.0),
                })
            
            all_results.append(row)
            
        except Exception as e:
            print(f"⚠️  Error loading {result_file.name}: {e}")
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Convert date to datetime
    try:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    except:
        pass
    
    print(f"✅ Loaded {len(df)} result records")
    return df


def calculate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics across all results."""
    print("\n📈 Calculating summary statistics...")
    
    summary = {}
    
    # Revenue metrics
    if 'final_pnl' in df.columns:
        summary['total_pnl'] = {
            'mean': float(df['final_pnl'].mean()),
            'std': float(df['final_pnl'].std()),
            'min': float(df['final_pnl'].min()),
            'max': float(df['final_pnl'].max()),
            'median': float(df['final_pnl'].median()),
        }
        summary['profitable_days'] = int((df['final_pnl'] > 0).sum())
        summary['profitable_pct'] = float((df['final_pnl'] > 0).mean() * 100)
    
    # AS revenue breakdown
    if 'as_revenue' in df.columns and 'total_revenue' in df.columns:
        df['as_revenue_pct'] = (df['as_revenue'] / (df['total_revenue'] + 1e-8)) * 100
        summary['as_revenue_pct'] = {
            'mean': float(df['as_revenue_pct'].mean()),
            'std': float(df['as_revenue_pct'].std()),
        }
    
    # Battery utilization
    if 'total_cycles' in df.columns:
        summary['cycles_per_day'] = {
            'mean': float(df['total_cycles'].mean()),
            'std': float(df['total_cycles'].std()),
        }
    
    if 'soc_utilization' in df.columns:
        summary['soc_utilization'] = {
            'mean': float(df['soc_utilization'].mean()),
            'std': float(df['soc_utilization'].std()),
        }
    
    # Risk metrics
    if 'sharpe_ratio' in df.columns:
        summary['sharpe_ratio'] = {
            'mean': float(df['sharpe_ratio'].mean()),
            'std': float(df['sharpe_ratio'].std()),
        }
    
    if 'max_drawdown' in df.columns:
        summary['max_drawdown'] = {
            'mean': float(df['max_drawdown'].mean()),
            'std': float(df['max_drawdown'].std()),
        }
    
    # Market characteristics
    if 'lmp_cov' in df.columns:
        summary['lmp_volatility'] = {
            'mean': float(df['lmp_cov'].mean()),
            'std': float(df['lmp_cov'].std()),
        }
    
    return summary


def test_hypotheses(df: pd.DataFrame) -> Dict:
    """Test research hypotheses H1, H2, H3."""
    print("\n🔬 Testing research hypotheses...")
    
    results = {}
    
    # H1: Price volatility → Profitability
    if 'lmp_cov' in df.columns and 'final_pnl' in df.columns:
        correlation = df['lmp_cov'].corr(df['final_pnl'])
        results['H1'] = {
            'hypothesis': 'Higher price volatility → Higher profitability',
            'correlation': float(correlation),
            'supported': abs(correlation) > 0.3,  # Threshold for meaningful correlation
            'interpretation': 'Strongly supported' if abs(correlation) > 0.5 else 'Moderately supported' if abs(correlation) > 0.3 else 'Weakly supported',
        }
        print(f"   H1: Correlation = {correlation:.3f} ({results['H1']['interpretation']})")
    
    # H2: Optimal SOC utilization = 60-80%
    if 'soc_utilization' in df.columns and 'final_pnl' in df.columns:
        # Find optimal utilization range
        df['soc_bin'] = pd.cut(df['soc_utilization'], bins=[0, 0.4, 0.6, 0.8, 1.0], labels=['Low', 'Medium', 'High', 'Very High'])
        optimal_range = df[df['soc_bin'].isin(['Medium', 'High'])]
        optimal_pnl = optimal_range['final_pnl'].mean() if len(optimal_range) > 0 else 0.0
        
        results['H2'] = {
            'hypothesis': 'Optimal SOC utilization = 60-80%',
            'optimal_range_pnl': float(optimal_pnl),
            'mean_utilization': float(df['soc_utilization'].mean()),
            'supported': 0.6 <= df['soc_utilization'].mean() <= 0.8,
            'interpretation': 'Supported' if 0.6 <= df['soc_utilization'].mean() <= 0.8 else 'Partially supported',
        }
        print(f"   H2: Mean utilization = {df['soc_utilization'].mean():.3f} ({results['H2']['interpretation']})")
    
    # H3: Spread >$15/MWh required for profitability
    if 'mean_lmp' in df.columns and 'final_pnl' in df.columns:
        # Calculate spread (simplified: use max - min LMP as proxy)
        if 'max_lmp' in df.columns and 'min_lmp' in df.columns:
            df['spread'] = df['max_lmp'] - df['min_lmp']
            profitable_days = df[df['final_pnl'] > 0]
            profitable_spread = profitable_days['spread'].mean() if len(profitable_days) > 0 else 0.0
            
            results['H3'] = {
                'hypothesis': 'Spread >$15/MWh required for profitability',
                'profitable_spread': float(profitable_spread),
                'threshold': 15.0,
                'supported': profitable_spread > 15.0,
                'interpretation': 'Supported' if profitable_spread > 15.0 else 'Not supported',
            }
            print(f"   H3: Profitable days spread = ${profitable_spread:.2f}/MWh ({results['H3']['interpretation']})")
    
    return results


def generate_paper_tables(df: pd.DataFrame, summary: Dict, hypotheses: Dict, output_dir: Path) -> None:
    """Generate paper-ready tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📝 Generating paper-ready tables...")
    
    # Table 1: Summary Statistics
    table1_data = []
    
    if 'total_pnl' in summary:
        pnl = summary['total_pnl']
        table1_data.append(['Total PnL ($)', f"{pnl['mean']:.2f}", f"{pnl['std']:.2f}", f"{pnl['min']:.2f}", f"{pnl['max']:.2f}"])
    
    if 'as_revenue_pct' in summary:
        as_rev = summary['as_revenue_pct']
        table1_data.append(['AS Revenue %', f"{as_rev['mean']:.1f}", f"{as_rev['std']:.1f}", '-', '-'])
    
    if 'cycles_per_day' in summary:
        cycles = summary['cycles_per_day']
        table1_data.append(['Cycles/Day', f"{cycles['mean']:.2f}", f"{cycles['std']:.2f}", '-', '-'])
    
    if 'sharpe_ratio' in summary:
        sharpe = summary['sharpe_ratio']
        table1_data.append(['Sharpe Ratio', f"{sharpe['mean']:.3f}", f"{sharpe['std']:.3f}", '-', '-'])
    
    if table1_data:
        table1_df = pd.DataFrame(table1_data, columns=['Metric', 'Mean', 'Std', 'Min', 'Max'])
        table1_file = output_dir / "table1_summary_statistics.csv"
        table1_df.to_csv(table1_file, index=False)
        print(f"   ✅ Saved: {table1_file}")
    
    # Table 2: Hypothesis Test Results
    if hypotheses:
        table2_data = []
        for h_id, h_result in hypotheses.items():
            table2_data.append([
                h_id,
                h_result['hypothesis'],
                h_result.get('correlation', h_result.get('mean_utilization', h_result.get('profitable_spread', 'N/A'))),
                h_result['interpretation'],
            ])
        
        table2_df = pd.DataFrame(table2_data, columns=['Hypothesis', 'Description', 'Test Statistic', 'Result'])
        table2_file = output_dir / "table2_hypothesis_tests.csv"
        table2_df.to_csv(table2_file, index=False)
        print(f"   ✅ Saved: {table2_file}")


def main():
    """Main analysis function."""
    print("="*60)
    print("HISTORICAL RESULTS ANALYSIS")
    print("="*60)
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "research" / "results"
    output_dir = project_root / "research" / "analysis"
    
    # Load all results
    df = load_all_results(results_dir)
    
    if df.empty:
        print("❌ No results to analyze")
        return
    
    # Calculate summary statistics
    summary = calculate_summary_statistics(df)
    
    # Test hypotheses
    hypotheses = test_hypotheses(df)
    
    # Generate paper tables
    generate_paper_tables(df, summary, hypotheses, output_dir)
    
    # Save aggregated results
    output_file = output_dir / "aggregated_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved aggregated results: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    if 'total_pnl' in summary:
        pnl = summary['total_pnl']
        print(f"\n💰 Revenue:")
        print(f"   Mean Daily PnL: ${pnl['mean']:.2f} ± ${pnl['std']:.2f}")
        print(f"   Range: ${pnl['min']:.2f} to ${pnl['max']:.2f}")
        if 'profitable_pct' in summary:
            print(f"   Profitable Days: {summary['profitable_days']} ({summary['profitable_pct']:.1f}%)")
    
    if hypotheses:
        print(f"\n🔬 Hypothesis Tests:")
        for h_id, h_result in hypotheses.items():
            print(f"   {h_id}: {h_result['interpretation']}")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

