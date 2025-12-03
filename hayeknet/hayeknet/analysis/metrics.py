"""Data analysis utilities for HayekNet simulation results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from hayeknet.analysis.results import ResultsReader, SimulationResults

# Set publication-quality plotting defaults
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
})


@dataclass
class AnalysisSummary:
    """Statistical summary of simulation results."""
    
    # Market statistics
    mean_load_mw: float
    std_load_mw: float
    mean_lmp_usd: float
    std_lmp_usd: float
    
    # Data assimilation performance
    enkf_mean_error: float
    enkf_rmse: float
    enkf_ensemble_spread: float
    
    # Bayesian inference
    mean_belief: float
    belief_entropy: float
    
    # RL performance
    mean_action_mw: float
    action_variance: float
    
    # Economic performance
    total_revenue_usd: float
    mean_profit_per_mw: float
    
    # Risk metrics
    value_at_risk_95: float
    conditional_var_95: float
    max_drawdown: float
    
    # Optional metrics
    sharpe_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_load_mw": self.mean_load_mw,
            "std_load_mw": self.std_load_mw,
            "mean_lmp_usd": self.mean_lmp_usd,
            "std_lmp_usd": self.std_lmp_usd,
            "enkf_mean_error": self.enkf_mean_error,
            "enkf_rmse": self.enkf_rmse,
            "enkf_ensemble_spread": self.enkf_ensemble_spread,
            "mean_belief": self.mean_belief,
            "belief_entropy": self.belief_entropy,
            "mean_action_mw": self.mean_action_mw,
            "action_variance": self.action_variance,
            "total_revenue_usd": self.total_revenue_usd,
            "mean_profit_per_mw": self.mean_profit_per_mw,
            "sharpe_ratio": self.sharpe_ratio or np.nan,
            "value_at_risk_95": self.value_at_risk_95,
            "conditional_var_95": self.conditional_var_95,
            "max_drawdown": self.max_drawdown,
        }


class ResultsAnalyzer:
    """Comprehensive analysis of simulation results."""
    
    def __init__(self, results: SimulationResults):
        """Initialize analyzer with simulation results."""
        self.results = results
        self.market_data = results.market_data
        self.mean_state = results.mean_state
        self.beliefs = results.beliefs
        self.rl_actions = np.array(results.rl_actions) if results.rl_actions else np.array([])
        self.option_prices = np.array(results.option_prices) if results.option_prices else np.array([])
    
    def compute_summary(self) -> AnalysisSummary:
        """Compute comprehensive statistical summary."""
        # Market statistics
        mean_load = float(self.market_data["net_load_mw"].mean())
        std_load = float(self.market_data["net_load_mw"].std())
        mean_lmp = float(self.market_data["lmp_usd"].mean())
        std_lmp = float(self.market_data["lmp_usd"].std())
        
        # EnKF performance
        enkf_mean_error = float(np.mean(np.abs(self.mean_state[:, 0] - self.market_data["net_load_mw"].values[:len(self.mean_state)])))
        enkf_rmse = float(np.sqrt(np.mean((self.mean_state[:, 0] - self.market_data["net_load_mw"].values[:len(self.mean_state)])**2)))
        enkf_spread = float(self.results.analysis_ensemble.std())
        
        # Bayesian metrics
        mean_belief = float(self.beliefs.mean())
        belief_entropy = self._compute_entropy(self.beliefs)
        
        # RL metrics
        mean_action = float(self.rl_actions.mean()) if len(self.rl_actions) > 0 else 0.0
        action_var = float(self.rl_actions.var()) if len(self.rl_actions) > 0 else 0.0
        
        # Economic metrics
        revenue_series = self._compute_revenue_series()
        total_revenue = float(revenue_series.sum())
        mean_profit = float(revenue_series.mean()) if len(revenue_series) > 0 else 0.0
        sharpe = self._compute_sharpe_ratio(revenue_series) if len(revenue_series) > 0 else None
        
        # Risk metrics
        var_95 = float(np.percentile(revenue_series, 5)) if len(revenue_series) > 0 else 0.0
        cvar_95 = float(revenue_series[revenue_series <= var_95].mean()) if len(revenue_series) > 0 else 0.0
        max_dd = self._compute_max_drawdown(revenue_series) if len(revenue_series) > 0 else 0.0
        
        return AnalysisSummary(
            mean_load_mw=mean_load,
            std_load_mw=std_load,
            mean_lmp_usd=mean_lmp,
            std_lmp_usd=std_lmp,
            enkf_mean_error=enkf_mean_error,
            enkf_rmse=enkf_rmse,
            enkf_ensemble_spread=enkf_spread,
            mean_belief=mean_belief,
            belief_entropy=belief_entropy,
            mean_action_mw=mean_action,
            action_variance=action_var,
            total_revenue_usd=total_revenue,
            mean_profit_per_mw=mean_profit,
            sharpe_ratio=sharpe,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            max_drawdown=max_dd,
        )
    
    @staticmethod
    def _compute_entropy(beliefs: np.ndarray) -> float:
        """Compute Shannon entropy of belief distribution."""
        # Treat beliefs as probabilities
        p = np.clip(beliefs, 1e-10, 1 - 1e-10)
        return float(-np.mean(p * np.log(p) + (1 - p) * np.log(1 - p)))
    
    def _compute_revenue_series(self) -> np.ndarray:
        """Compute revenue time series from actions and LMPs."""
        if len(self.rl_actions) == 0:
            return np.array([])
        
        min_len = min(len(self.rl_actions), len(self.market_data))
        actions = self.rl_actions[:min_len]
        lmps = self.market_data["lmp_usd"].values[:min_len]
        
        # Revenue = action * LMP (assuming all bids clear)
        return actions * lmps
    
    @staticmethod
    def _compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Compute Sharpe ratio of returns."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return float((returns.mean() - risk_free_rate) / returns.std())
    
    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        """Compute maximum drawdown from cumulative returns."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return float(drawdown.max())
    
    def plot_market_overview(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot market data overview."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Load plot
        axes[0].plot(self.market_data.index, self.market_data["net_load_mw"], 
                    label="Net Load", color="steelblue", linewidth=1)
        axes[0].set_ylabel("Load (MW)")
        axes[0].set_title("ERCOT Market Conditions")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # LMP plot
        axes[1].plot(self.market_data.index, self.market_data["lmp_usd"], 
                    label="LMP", color="darkorange", linewidth=1)
        axes[1].set_ylabel("LMP (USD/MWh)")
        axes[1].set_xlabel("Time Step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_enkf_performance(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot EnKF state estimation performance."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        min_len = min(len(self.mean_state), len(self.market_data))
        timesteps = np.arange(min_len)
        true_load = self.market_data["net_load_mw"].values[:min_len]
        estimated_load = self.mean_state[:min_len, 0]
        
        # State estimate vs true
        axes[0].plot(timesteps, true_load, label="True Load", color="black", linewidth=1.5, alpha=0.7)
        axes[0].plot(timesteps, estimated_load, label="EnKF Estimate", color="crimson", linewidth=1, linestyle="--")
        axes[0].set_ylabel("Load (MW)")
        axes[0].set_title("Data Assimilation Performance")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = true_load - estimated_load
        axes[1].plot(timesteps, residuals, label="Residuals", color="purple", linewidth=0.8)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        axes[1].fill_between(timesteps, residuals, 0, alpha=0.3, color="purple")
        axes[1].set_ylabel("Error (MW)")
        axes[1].set_xlabel("Time Step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_belief_evolution(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot Bayesian belief evolution."""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        timesteps = np.arange(len(self.beliefs))
        ax.plot(timesteps, self.beliefs, label="P(High Demand)", color="teal", linewidth=1.5)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Prior")
        ax.fill_between(timesteps, 0, self.beliefs, alpha=0.2, color="teal")
        
        ax.set_ylabel("Belief Probability")
        ax.set_xlabel("Time Step")
        ax.set_title("Bayesian Belief Update Evolution")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_rl_actions(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot RL bidding actions."""
        if len(self.rl_actions) == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No RL actions recorded", ha="center", va="center")
            return fig
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        timesteps = np.arange(len(self.rl_actions))
        
        # Actions over time
        axes[0].plot(timesteps, self.rl_actions, label="Bid (MW)", color="darkgreen", linewidth=1)
        axes[0].set_ylabel("Bid (MW)")
        axes[0].set_title("RL Agent Bidding Strategy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Action distribution
        axes[1].hist(self.rl_actions, bins=30, color="darkgreen", alpha=0.7, edgecolor="black")
        axes[1].set_ylabel("Frequency")
        axes[1].set_xlabel("Bid (MW)")
        axes[1].set_title("Bid Distribution")
        axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def plot_economic_performance(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot economic performance metrics."""
        revenue = self._compute_revenue_series()
        
        if len(revenue) == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No revenue data available", ha="center", va="center")
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        timesteps = np.arange(len(revenue))
        cumulative_revenue = np.cumsum(revenue)
        
        # Cumulative revenue
        axes[0, 0].plot(timesteps, cumulative_revenue, color="forestgreen", linewidth=1.5)
        axes[0, 0].set_ylabel("Cumulative Revenue (USD)")
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_title("Cumulative Revenue")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Revenue distribution
        axes[0, 1].hist(revenue, bins=30, color="forestgreen", alpha=0.7, edgecolor="black")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_xlabel("Revenue per Step (USD)")
        axes[0, 1].set_title("Revenue Distribution")
        axes[0, 1].grid(True, alpha=0.3, axis="y")
        
        # Drawdown
        running_max = np.maximum.accumulate(cumulative_revenue)
        drawdown = running_max - cumulative_revenue
        axes[1, 0].fill_between(timesteps, 0, drawdown, color="red", alpha=0.5)
        axes[1, 0].set_ylabel("Drawdown (USD)")
        axes[1, 0].set_xlabel("Time Step")
        axes[1, 0].set_title(f"Drawdown (Max: ${drawdown.max():.2f})")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        stats.probplot(revenue, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Revenue Normality Check")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def generate_full_report(self, output_dir: Path) -> AnalysisSummary:
        """Generate complete analysis report with all plots."""
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute summary statistics
        summary = self.compute_summary()
        
        # Generate all plots
        print("📊 Generating market overview...")
        self.plot_market_overview(output_dir / "01_market_overview.png")
        
        print("📊 Generating EnKF performance...")
        self.plot_enkf_performance(output_dir / "02_enkf_performance.png")
        
        print("📊 Generating belief evolution...")
        self.plot_belief_evolution(output_dir / "03_belief_evolution.png")
        
        if len(self.rl_actions) > 0:
            print("📊 Generating RL actions...")
            self.plot_rl_actions(output_dir / "04_rl_actions.png")
        
        revenue = self._compute_revenue_series()
        if len(revenue) > 0:
            print("📊 Generating economic performance...")
            self.plot_economic_performance(output_dir / "05_economic_performance.png")
        
        # Write summary to JSON
        import json
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        print(f"✅ Full report generated in {output_dir}")
        
        return summary


def compare_runs(
    run_ids: List[str],
    results_dir: Path,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compare multiple simulation runs."""
    reader = ResultsReader(results_dir)
    
    summaries = []
    for run_id in run_ids:
        results = reader.read(run_id)
        analyzer = ResultsAnalyzer(results)
        summary = analyzer.compute_summary()
        
        summary_dict = summary.to_dict()
        summary_dict["run_id"] = run_id
        summaries.append(summary_dict)
    
    comparison_df = pd.DataFrame(summaries)
    
    if output_path:
        comparison_df.to_csv(output_path, index=False)
        print(f"✅ Comparison saved to {output_path}")
    
    return comparison_df
