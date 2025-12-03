#!/usr/bin/env python3
"""Generate publication-quality visualizations for research paper.

Creates:
1. SOC trajectories over time (SCED vs RTC+B)
2. Revenue comparison (stacked: energy + AS)
3. Price formation (LMP with ASDC adjustments)
4. AS participation heatmap
5. Sharpe ratio comparison
6. Cumulative PnL curves
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_comparison_data(comparison_dir: Path) -> pd.DataFrame:
    """Load SCED vs RTC+B comparison results."""
    results_file = comparison_dir / "rtcb_comparison_results.csv"
    
    if not results_file.exists():
        print(f"⚠️  Comparison results not found: {results_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(results_file)
    print(f"✅ Loaded {len(df)} comparison records")
    return df


def load_historical_data(analysis_dir: Path) -> pd.DataFrame:
    """Load aggregated historical results."""
    results_file = analysis_dir / "aggregated_results.csv"
    
    if not results_file.exists():
        print(f"⚠️  Historical results not found: {results_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(results_file)
    print(f"✅ Loaded {len(df)} historical records")
    return df


def plot_soc_trajectories(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot SOC trajectories for SCED vs RTC+B."""
    if comparison_df.empty:
        print("⚠️  No comparison data for SOC trajectories")
        return
    
    print("📊 Plotting SOC trajectories...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for variant in comparison_df['variant'].unique():
        variant_data = comparison_df[comparison_df['variant'] == variant]
        ax.plot(
            variant_data.index,
            variant_data['soc'] * 100,
            label=variant,
            alpha=0.7,
            linewidth=1.5
        )
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('State of Charge (%)', fontsize=12)
    ax.set_title('Battery SOC Trajectories: SCED vs RTC+B', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_file = output_dir / "soc_trajectories.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")


def plot_revenue_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot stacked revenue comparison (energy + AS)."""
    if comparison_df.empty:
        print("⚠️  No comparison data for revenue comparison")
        return
    
    print("📊 Plotting revenue comparison...")
    
    # Aggregate by variant
    revenue_summary = comparison_df.groupby('variant').agg({
        'energy_revenue': 'sum',
        'as_revenue': 'sum',
        'total_revenue': 'sum',
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(revenue_summary))
    width = 0.6
    
    ax.bar(x, revenue_summary['energy_revenue'], width, label='Energy Revenue', alpha=0.8)
    ax.bar(x, revenue_summary['as_revenue'], width, bottom=revenue_summary['energy_revenue'],
           label='AS Revenue', alpha=0.8)
    
    ax.set_xlabel('Market Design', fontsize=12)
    ax.set_ylabel('Total Revenue ($)', fontsize=12)
    ax.set_title('Revenue Breakdown: Energy vs Ancillary Services', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(revenue_summary['variant'], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "revenue_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")


def plot_cumulative_pnl(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot cumulative PnL curves."""
    if comparison_df.empty:
        print("⚠️  No comparison data for cumulative PnL")
        return
    
    print("📊 Plotting cumulative PnL...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for variant in comparison_df['variant'].unique():
        variant_data = comparison_df[comparison_df['variant'] == variant].copy()
        variant_data['cumulative_pnl'] = variant_data['total_revenue'].cumsum()
        
        ax.plot(
            variant_data.index,
            variant_data['cumulative_pnl'],
            label=variant,
            linewidth=2,
            alpha=0.8
        )
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=12)
    ax.set_title('Cumulative Revenue: SCED vs RTC+B', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    output_file = output_dir / "cumulative_pnl.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")


def plot_sharpe_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Sharpe ratio comparison."""
    if summary_df.empty or 'sharpe_ratio' not in summary_df.columns:
        print("⚠️  No summary data for Sharpe ratio comparison")
        return
    
    print("📊 Plotting Sharpe ratio comparison...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(summary_df['variant'], summary_df['sharpe_ratio'], alpha=0.8)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Risk-Adjusted Returns: Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(summary_df['variant'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "sharpe_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")


def plot_as_participation_heatmap(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot AS participation heatmap."""
    if comparison_df.empty:
        print("⚠️  No comparison data for AS participation heatmap")
        return
    
    print("📊 Plotting AS participation heatmap...")
    
    # Calculate AS participation by variant and time
    as_cols = ['reg_up_cleared_mw', 'reg_down_cleared_mw', 'rrs_cleared_mw', 'ecrs_cleared_mw']
    
    heatmap_data = []
    for variant in comparison_df['variant'].unique():
        variant_data = comparison_df[comparison_df['variant'] == variant]
        for col in as_cols:
            if col in variant_data.columns:
                total_as = variant_data[col].sum()
                heatmap_data.append({
                    'variant': variant,
                    'service': col.replace('_cleared_mw', '').replace('_', ' ').title(),
                    'total_mw': total_as
                })
    
    if not heatmap_data:
        print("   ⚠️  No AS data available")
        return
    
    heatmap_df = pd.DataFrame(heatmap_data)
    pivot = heatmap_df.pivot(index='variant', columns='service', values='total_mw')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Total MW'})
    ax.set_title('AS Participation by Market Design', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ancillary Service', fontsize=12)
    ax.set_ylabel('Market Design', fontsize=12)
    
    plt.tight_layout()
    output_file = output_dir / "as_participation_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")


def plot_historical_trends(historical_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot historical trends over time."""
    if historical_df.empty:
        print("⚠️  No historical data for trends")
        return
    
    print("📊 Plotting historical trends...")
    
    # Convert date to datetime if possible
    if 'date' in historical_df.columns:
        try:
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            historical_df = historical_df.sort_values('date')
        except:
            pass
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Daily PnL over time
    if 'final_pnl' in historical_df.columns:
        axes[0, 0].plot(historical_df.index, historical_df['final_pnl'], alpha=0.7, linewidth=1.5)
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axes[0, 0].set_ylabel('Daily PnL ($)', fontsize=11)
        axes[0, 0].set_title('Daily Profitability Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
    
    # SOC utilization over time
    if 'soc_utilization' in historical_df.columns:
        axes[0, 1].plot(historical_df.index, historical_df['soc_utilization'], alpha=0.7, linewidth=1.5)
        axes[0, 1].set_ylabel('SOC Utilization', fontsize=11)
        axes[0, 1].set_title('Battery Utilization Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Cycles per day
    if 'total_cycles' in historical_df.columns:
        axes[1, 0].plot(historical_df.index, historical_df['total_cycles'], alpha=0.7, linewidth=1.5)
        axes[1, 0].set_ylabel('Cycles/Day', fontsize=11)
        axes[1, 0].set_title('Battery Cycling Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Sharpe ratio over time
    if 'sharpe_ratio' in historical_df.columns:
        axes[1, 1].plot(historical_df.index, historical_df['sharpe_ratio'], alpha=0.7, linewidth=1.5)
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axes[1, 1].set_ylabel('Sharpe Ratio', fontsize=11)
        axes[1, 1].set_title('Risk-Adjusted Returns Over Time', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "historical_trends.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")


def main():
    """Main visualization generation function."""
    print("="*60)
    print("GENERATING PAPER VISUALIZATIONS")
    print("="*60)
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    comparison_dir = project_root / "research" / "rtcb_evaluation"
    analysis_dir = project_root / "research" / "analysis"
    output_dir = project_root / "research" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    comparison_df = load_comparison_data(comparison_dir)
    historical_df = load_historical_data(analysis_dir)
    
    # Load summary if available
    summary_file = comparison_dir / "rtcb_comparison_summary.csv"
    summary_df = pd.DataFrame()
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
    
    # Generate plots
    if not comparison_df.empty:
        plot_soc_trajectories(comparison_df, output_dir)
        plot_revenue_comparison(comparison_df, output_dir)
        plot_cumulative_pnl(comparison_df, output_dir)
        plot_as_participation_heatmap(comparison_df, output_dir)
    
    if not summary_df.empty:
        plot_sharpe_comparison(summary_df, output_dir)
    
    if not historical_df.empty:
        plot_historical_trends(historical_df, output_dir)
    
    print("\n" + "="*60)
    print("✅ VISUALIZATION GENERATION COMPLETE")
    print("="*60)
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

