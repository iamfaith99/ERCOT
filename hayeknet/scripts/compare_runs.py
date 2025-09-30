#!/usr/bin/env python3
"""Compare multiple simulation runs and generate comparative analysis."""

import argparse
from pathlib import Path

from python.analysis import compare_runs
from python.results import ResultsReader


def main() -> None:
    """Compare multiple simulation runs."""
    parser = argparse.ArgumentParser(description="Compare HayekNet simulation runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Specific run IDs to compare (default: all runs)",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=5,
        help="Number of latest runs to compare (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: runs/comparison_results.csv)",
    )
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "runs"
    
    # Determine which runs to compare
    reader = ResultsReader(results_dir)
    all_runs = reader.list_runs()
    
    if not all_runs:
        print("❌ No simulation runs found in", results_dir)
        print("   Run 'make run-analysis' first to generate results")
        return
    
    if args.runs:
        # Use specified runs
        run_ids = args.runs
        # Validate they exist
        for run_id in run_ids:
            if run_id not in all_runs:
                print(f"⚠️  Warning: Run {run_id} not found, skipping")
        run_ids = [r for r in run_ids if r in all_runs]
    else:
        # Use latest N runs
        run_ids = all_runs[-args.latest:]
    
    if len(run_ids) < 2:
        print("❌ Need at least 2 runs for comparison")
        print(f"   Found: {len(run_ids)} valid runs")
        return
    
    print(f"📊 Comparing {len(run_ids)} simulation runs")
    print("=" * 60)
    for run_id in run_ids:
        print(f"  - {run_id}")
    print("=" * 60)
    
    # Set output path
    output_path = args.output or (results_dir / "comparison_results.csv")
    
    # Perform comparison
    comparison_df = compare_runs(run_ids, results_dir, output_path)
    
    # Display summary
    print("\n" + "=" * 60)
    print("📈 Comparison Summary")
    print("=" * 60)
    
    # Key metrics table
    key_metrics = [
        'run_id',
        'mean_load_mw',
        'mean_lmp_usd',
        'enkf_rmse',
        'mean_belief',
        'mean_action_mw',
        'total_revenue_usd',
        'sharpe_ratio',
        'max_drawdown',
    ]
    
    display_df = comparison_df[key_metrics].copy()
    display_df['run_id'] = display_df['run_id'].str[:12]  # Truncate for display
    
    print("\nKey Metrics:")
    print(display_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Rankings
    print("\n" + "=" * 60)
    print("🏆 Performance Rankings")
    print("=" * 60)
    
    best_revenue_idx = comparison_df['total_revenue_usd'].idxmax()
    print(f"\n1. Best Revenue: {comparison_df.loc[best_revenue_idx, 'run_id']}")
    print(f"   Total: ${comparison_df.loc[best_revenue_idx, 'total_revenue_usd']:,.2f}")
    
    if comparison_df['sharpe_ratio'].notna().any():
        best_sharpe_idx = comparison_df['sharpe_ratio'].idxmax()
        print(f"\n2. Best Risk-Adjusted Returns: {comparison_df.loc[best_sharpe_idx, 'run_id']}")
        print(f"   Sharpe: {comparison_df.loc[best_sharpe_idx, 'sharpe_ratio']:.3f}")
    
    best_enkf_idx = comparison_df['enkf_rmse'].idxmin()
    print(f"\n3. Best State Estimation: {comparison_df.loc[best_enkf_idx, 'run_id']}")
    print(f"   RMSE: {comparison_df.loc[best_enkf_idx, 'enkf_rmse']:.2f} MW")
    
    print("\n" + "=" * 60)
    print(f"✅ Comparison saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
