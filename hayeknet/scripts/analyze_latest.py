#!/usr/bin/env python3
"""Generate analysis report for the latest simulation run."""

from pathlib import Path

from hayeknet.analysis.metrics import ResultsAnalyzer
from hayeknet.analysis.results import ResultsReader


def main() -> None:
    """Analyze the most recent simulation run."""
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "runs"
    
    # Find latest run
    reader = ResultsReader(results_dir)
    runs = reader.list_runs()
    
    if not runs:
        print("❌ No simulation runs found in", results_dir)
        print("   Run 'make run-analysis' first to generate results")
        return
    
    latest_run = runs[-1]
    print(f"📊 Analyzing latest run: {latest_run}")
    print("=" * 60)
    
    # Load and analyze
    results = reader.read(latest_run)
    analyzer = ResultsAnalyzer(results)
    
    # Generate report
    output_dir = results_dir / latest_run / "analysis"
    summary = analyzer.generate_full_report(output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 Analysis Summary")
    print("=" * 60)
    print(f"Market Conditions:")
    print(f"  Mean Load: {summary.mean_load_mw:.0f} MW")
    print(f"  Std Load: {summary.std_load_mw:.0f} MW")
    print(f"  Mean LMP: ${summary.mean_lmp_usd:.2f}/MWh")
    print(f"  Std LMP: ${summary.std_lmp_usd:.2f}/MWh")
    
    print(f"\nData Assimilation:")
    print(f"  EnKF RMSE: {summary.enkf_rmse:.2f} MW")
    print(f"  Ensemble Spread: {summary.enkf_ensemble_spread:.2f}")
    
    print(f"\nBayesian Reasoning:")
    print(f"  Mean Belief: {summary.mean_belief:.3f}")
    print(f"  Belief Entropy: {summary.belief_entropy:.3f}")
    
    print(f"\nRL Performance:")
    print(f"  Mean Action: {summary.mean_action_mw:.2f} MW")
    print(f"  Action Variance: {summary.action_variance:.2f}")
    
    print(f"\nEconomic Performance:")
    print(f"  Total Revenue: ${summary.total_revenue_usd:,.2f}")
    print(f"  Mean Profit/MW: ${summary.mean_profit_per_mw:.2f}")
    if summary.sharpe_ratio:
        print(f"  Sharpe Ratio: {summary.sharpe_ratio:.3f}")
    
    print(f"\nRisk Metrics:")
    print(f"  VaR (95%): ${summary.value_at_risk_95:.2f}")
    print(f"  CVaR (95%): ${summary.conditional_var_95:.2f}")
    print(f"  Max Drawdown: ${summary.max_drawdown:.2f}")
    
    print("\n" + "=" * 60)
    print(f"✅ Full report saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
