#!/usr/bin/env python3
"""Example: Run a complete HayekNet experiment with multiple configurations."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from python.main_with_analysis import run_simulation
from python.analysis import compare_runs
from python.results import ResultsReader


def run_ensemble_size_experiment() -> List[str]:
    """
    Experiment: Effect of ensemble size on EnKF performance.
    
    Tests ensemble sizes: [25, 50, 100, 200]
    """
    print("=" * 60)
    print("🧪 Experiment: Ensemble Size Effect")
    print("=" * 60)
    
    ensemble_sizes = [25, 50, 100, 200]
    run_ids = []
    
    for size in ensemble_sizes:
        print(f"\n▶️  Testing ensemble_size={size}")
        results, run_dir = run_simulation(
            persist_results=True,
            generate_analysis=True,
            ensemble_size=size,
            rl_timesteps=2_000,
            seed=42,  # Same seed for fair comparison
        )
        if run_dir:
            run_ids.append(run_dir.name)
    
    return run_ids


def run_rl_timesteps_experiment() -> List[str]:
    """
    Experiment: Effect of RL training duration on performance.
    
    Tests timesteps: [1_000, 2_000, 5_000, 10_000]
    """
    print("=" * 60)
    print("🧪 Experiment: RL Training Duration")
    print("=" * 60)
    
    timesteps = [1_000, 2_000, 5_000, 10_000]
    run_ids = []
    
    for ts in timesteps:
        print(f"\n▶️  Testing rl_timesteps={ts}")
        results, run_dir = run_simulation(
            persist_results=True,
            generate_analysis=True,
            ensemble_size=100,
            rl_timesteps=ts,
            seed=42,
        )
        if run_dir:
            run_ids.append(run_dir.name)
    
    return run_ids


def run_seed_robustness_experiment() -> List[str]:
    """
    Experiment: Robustness across different random seeds.
    
    Tests 5 different random seeds: [42, 123, 456, 789, 2025]
    """
    print("=" * 60)
    print("🧪 Experiment: Random Seed Robustness")
    print("=" * 60)
    
    seeds = [42, 123, 456, 789, 2025]
    run_ids = []
    
    for seed in seeds:
        print(f"\n▶️  Testing seed={seed}")
        results, run_dir = run_simulation(
            persist_results=True,
            generate_analysis=True,
            ensemble_size=100,
            rl_timesteps=2_000,
            seed=seed,
        )
        if run_dir:
            run_ids.append(run_dir.name)
    
    return run_ids


def analyze_experiment(experiment_name: str, run_ids: List[str]) -> None:
    """Generate comparative analysis for experiment."""
    print("\n" + "=" * 60)
    print(f"📊 Analyzing Experiment: {experiment_name}")
    print("=" * 60)
    
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "runs"
    output_path = results_dir / f"{experiment_name}_comparison.csv"
    
    # Generate comparison
    comparison_df = compare_runs(run_ids, results_dir, output_path)
    
    # Display key results
    print("\nKey Metrics:")
    key_cols = ['run_id', 'enkf_rmse', 'mean_belief', 'total_revenue_usd', 'sharpe_ratio']
    print(comparison_df[key_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"  Mean Revenue: ${comparison_df['total_revenue_usd'].mean():,.2f}")
    print(f"  Std Revenue: ${comparison_df['total_revenue_usd'].std():,.2f}")
    print(f"  Best Revenue: ${comparison_df['total_revenue_usd'].max():,.2f}")
    print(f"  Worst Revenue: ${comparison_df['total_revenue_usd'].min():,.2f}")
    
    if comparison_df['sharpe_ratio'].notna().any():
        print(f"  Mean Sharpe: {comparison_df['sharpe_ratio'].mean():.4f}")
        print(f"  Best Sharpe: {comparison_df['sharpe_ratio'].max():.4f}")


def main() -> None:
    """Run all experiments."""
    print("🚀 HayekNet Experimental Suite")
    print("=" * 60)
    print("Running systematic experiments to analyze:")
    print("  1. Ensemble size effect on DA performance")
    print("  2. RL training duration effect on revenue")
    print("  3. Robustness to random initialization")
    print()
    input("Press Enter to start experiments (this will take ~10-20 minutes)...")
    
    # Experiment 1: Ensemble Size
    print("\n\n")
    ensemble_runs = run_ensemble_size_experiment()
    analyze_experiment("ensemble_size_effect", ensemble_runs)
    
    # Experiment 2: RL Timesteps
    print("\n\n")
    rl_runs = run_rl_timesteps_experiment()
    analyze_experiment("rl_duration_effect", rl_runs)
    
    # Experiment 3: Seed Robustness
    print("\n\n")
    seed_runs = run_seed_robustness_experiment()
    analyze_experiment("seed_robustness", seed_runs)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 All Experiments Complete!")
    print("=" * 60)
    print(f"Total runs: {len(ensemble_runs) + len(rl_runs) + len(seed_runs)}")
    print(f"\nResults saved in: runs/")
    print(f"Comparison CSVs:")
    print(f"  - runs/ensemble_size_effect_comparison.csv")
    print(f"  - runs/rl_duration_effect_comparison.csv")
    print(f"  - runs/seed_robustness_comparison.csv")
    print("\nNext steps:")
    print("  1. Review comparison CSVs for detailed metrics")
    print("  2. Open Quarto notebooks for rich visualizations:")
    print("     cd docs && quarto render comparative_analysis.qmd")
    print("  3. Generate custom analyses using Python API")


if __name__ == "__main__":
    main()
