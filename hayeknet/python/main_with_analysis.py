"""Enhanced entry point with results persistence and analysis generation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

# Import juliacall FIRST to prevent segfaults with torch
try:
    from juliacall import Main as jl
except ImportError:
    jl = None

import numpy as np

from envs.rtc_env import RTCEnv, RTCEnvConfig
from python.agents import BayesianReasoner, RLTrainer, decide_bid
from python.analysis import ResultsAnalyzer
from python.data import ERCOTDataClient, build_observation_operator
from python.results import ResultsWriter, SimulationMetadata, SimulationResults
from python.utils import init_julia, price_option, run_enkf, validate_constraints


def run_simulation(
    *,
    persist_results: bool = True,
    generate_analysis: bool = True,
    results_dir: Optional[Path] = None,
    ensemble_size: int = 100,
    rl_timesteps: int = 2_000,
    rl_vector_envs: int = 2,
    seed: Optional[int] = None,
) -> tuple[SimulationResults, Path | None]:
    """
    Run complete HayekNet simulation with optional persistence and analysis.
    
    Parameters
    ----------
    persist_results : bool
        Whether to persist results to disk
    generate_analysis : bool
        Whether to generate analysis plots and summary
    results_dir : Path, optional
        Directory for storing results (default: repo_root/runs)
    ensemble_size : int
        Number of ensemble members for EnKF
    rl_timesteps : int
        Total timesteps for RL training
    rl_vector_envs : int
        Number of vectorized environments for RL
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    results : SimulationResults
        Complete simulation results
    run_dir : Path | None
        Directory where results were saved (if persist_results=True)
    """
    start_time = time.time()
    
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize
    repo_root = Path(__file__).resolve().parents[1]
    init_julia(repo_root)
    
    if results_dir is None:
        results_dir = repo_root / "runs"
    
    print("=" * 60)
    print("🚀 HayekNet Simulation Starting")
    print("=" * 60)
    
    # Capture metadata
    config = {
        "ensemble_size": ensemble_size,
        "rl_timesteps": rl_timesteps,
        "rl_vector_envs": rl_vector_envs,
        "seed": seed,
    }
    metadata = SimulationMetadata.capture(config)
    print(f"📋 Run ID: {metadata.run_id}")
    print(f"🔧 Git SHA: {metadata.git_sha}")
    print(f"🌿 Git Branch: {metadata.git_branch}")
    print(f"🐍 Python: {metadata.python_version}")
    print(f"💎 Julia: {metadata.julia_version}")
    
    # ========================================================================
    # PHASE 1: Data Ingestion
    # ========================================================================
    print("\n" + "=" * 60)
    print("📊 PHASE 1: Data Ingestion")
    print("=" * 60)
    
    data_client = ERCOTDataClient()
    df, obs_matrix = data_client.fetch_rtc_like()
    
    print(f"✅ Fetched {len(df)} data points")
    print(f"   Load range: {df['net_load_mw'].min():.0f} - {df['net_load_mw'].max():.0f} MW")
    print(f"   LMP range: ${df['lmp_usd'].min():.2f} - ${df['lmp_usd'].max():.2f}/MWh")
    
    # ========================================================================
    # PHASE 2: Data Assimilation (EnKF)
    # ========================================================================
    print("\n" + "=" * 60)
    print("🔬 PHASE 2: Data Assimilation (EnKF)")
    print("=" * 60)
    
    state_dim = obs_matrix.shape[0]
    H = build_observation_operator(obs_matrix)
    R = np.eye(state_dim) * 15.0
    
    prior_state = np.repeat(obs_matrix[:, :1], ensemble_size, axis=1)
    perturbations = np.random.normal(0, 50, size=prior_state.shape)
    prior_ensemble = prior_state + perturbations
    
    obs_ensemble = np.repeat(obs_matrix[:, :1], ensemble_size, axis=1)
    mean_state, analysis_ensemble = run_enkf(
        prior_ensemble, obs_ensemble, H, R, inflation=1.05
    )
    
    enkf_rmse = np.sqrt(np.mean((mean_state[0] - obs_matrix[0, 0])**2))
    print(f"✅ EnKF update complete")
    print(f"   Mean state: {mean_state}")
    print(f"   RMSE: {enkf_rmse:.2f} MW")
    print(f"   Ensemble spread: {analysis_ensemble.std():.2f}")
    
    enkf_metrics = {
        "rmse": float(enkf_rmse),
        "ensemble_spread": float(analysis_ensemble.std()),
        "ensemble_size": ensemble_size,
    }
    
    # ========================================================================
    # PHASE 3: Bayesian Reasoning
    # ========================================================================
    print("\n" + "=" * 60)
    print("🧠 PHASE 3: Bayesian Reasoning")
    print("=" * 60)
    
    reasoner = BayesianReasoner(prior_high=0.35)
    belief = reasoner.update(float(mean_state[0]))
    beliefs = np.full(len(df), belief, dtype=float)
    
    print(f"✅ Bayesian update complete")
    print(f"   Prior P(high): 0.35")
    print(f"   Posterior P(high): {belief:.3f}")
    print(f"   Evidence: {mean_state[0]:.2f} MW")
    
    bayesian_metrics = {
        "prior_high": 0.35,
        "posterior_high": float(belief),
        "evidence": float(mean_state[0]),
    }
    
    # ========================================================================
    # PHASE 4: Reinforcement Learning
    # ========================================================================
    print("\n" + "=" * 60)
    print("🤖 PHASE 4: Reinforcement Learning")
    print("=" * 60)
    
    config_obj = RTCEnvConfig()
    
    rl_actions = []
    rl_info = {}
    model = None
    
    try:
        trainer = RLTrainer(vector_envs=rl_vector_envs, total_timesteps=rl_timesteps)
        model, rl_info = trainer.train(lambda: RTCEnv(df, beliefs, config_obj))
        
        print(f"✅ RL training complete")
        print(f"   Timesteps: {rl_info.get('timesteps', 'unknown')}")
        
        # Generate actions for entire episode
        test_env = RTCEnv(df, beliefs, config_obj)
        observation, _ = test_env.reset()
        
        for _ in range(len(df)):
            action = decide_bid(model, observation)
            rl_actions.append(float(action))
            observation, _, terminated, truncated, _ = test_env.step([action])
            if terminated or truncated:
                break
        
        print(f"   Generated {len(rl_actions)} actions")
        print(f"   Mean bid: {np.mean(rl_actions):.2f} MW")
        print(f"   Bid std: {np.std(rl_actions):.2f} MW")
        
    except RuntimeError as exc:
        print(f"⚠️  RL training failed: {exc}")
        rl_actions = [float(config_obj.max_bid_mw / 2)] * len(df)
        rl_info = {"warning": str(exc)}
    
    # ========================================================================
    # PHASE 5: Option Pricing
    # ========================================================================
    print("\n" + "=" * 60)
    print("💰 PHASE 5: Option Pricing")
    print("=" * 60)
    
    option_prices = []
    for i in range(min(10, len(df))):  # Price first 10 timesteps as example
        hedge_price = price_option(
            float(df["lmp_usd"].iloc[i]),
            strike=float(df["lmp_usd"].quantile(0.9)),
            rate=0.05,
            volatility=0.2,
            maturity=0.25,
            steps=32,
            trajectories=256,
        )
        option_prices.append(hedge_price)
    
    print(f"✅ Option pricing complete")
    print(f"   Priced {len(option_prices)} options")
    print(f"   Mean hedge price: ${np.mean(option_prices):.2f}")
    print(f"   Strike: ${df['lmp_usd'].quantile(0.9):.2f}/MWh")
    
    # ========================================================================
    # PHASE 6: Constraint Validation
    # ========================================================================
    print("\n" + "=" * 60)
    print("✓ PHASE 6: Constraint Validation")
    print("=" * 60)
    
    edges = [[1, 2], [2, 3], [3, 4]]
    constraints_satisfied = []
    
    for action in rl_actions[:min(100, len(rl_actions))]:
        satisfied = validate_constraints(edges, 1, 4, capacity=200.0, bid=action)
        constraints_satisfied.append(satisfied)
    
    satisfaction_rate = np.mean(constraints_satisfied) if constraints_satisfied else 0.0
    print(f"✅ Constraint validation complete")
    print(f"   Checked {len(constraints_satisfied)} bids")
    print(f"   Satisfaction rate: {satisfaction_rate:.1%}")
    
    # ========================================================================
    # PHASE 7: Economic Analysis
    # ========================================================================
    print("\n" + "=" * 60)
    print("💵 PHASE 7: Economic Analysis")
    print("=" * 60)
    
    min_len = min(len(rl_actions), len(df))
    revenue = np.array(rl_actions[:min_len]) * df["lmp_usd"].values[:min_len]
    total_revenue = float(revenue.sum())
    mean_revenue = float(revenue.mean())
    
    print(f"✅ Economic analysis complete")
    print(f"   Total revenue: ${total_revenue:,.2f}")
    print(f"   Mean revenue/step: ${mean_revenue:.2f}")
    print(f"   Revenue std: ${revenue.std():.2f}")
    
    economic_metrics = {
        "total_revenue": total_revenue,
        "mean_revenue": mean_revenue,
        "revenue_std": float(revenue.std()),
    }
    
    # ========================================================================
    # Wrap Up
    # ========================================================================
    wall_time = time.time() - start_time
    metadata.wall_time_seconds = wall_time
    
    print("\n" + "=" * 60)
    print("✅ HayekNet Simulation Complete")
    print("=" * 60)
    print(f"⏱️  Wall time: {wall_time:.2f} seconds")
    
    # Build results object
    results = SimulationResults(
        mean_state=mean_state,
        analysis_ensemble=analysis_ensemble,
        belief_trace=beliefs,
        rl_actions=rl_actions,
        option_prices=option_prices,
        constraints_satisfied=constraints_satisfied,
        market_data=df,
        beliefs=beliefs,
        rl_metadata=rl_info,
        enkf_metrics=enkf_metrics,
        bayesian_metrics=bayesian_metrics,
        economic_metrics=economic_metrics,
        metadata=metadata,
    )
    
    run_dir = None
    
    # Persist results
    if persist_results:
        print("\n" + "=" * 60)
        print("💾 Persisting Results")
        print("=" * 60)
        
        writer = ResultsWriter(results_dir=results_dir)
        run_dir = writer.write(results)
    
    # Generate analysis
    if generate_analysis and run_dir is not None:
        print("\n" + "=" * 60)
        print("📈 Generating Analysis")
        print("=" * 60)
        
        analyzer = ResultsAnalyzer(results)
        summary = analyzer.generate_full_report(run_dir / "analysis")
        
        print("\n📊 Summary Statistics:")
        print(f"   Mean Load: {summary.mean_load_mw:.0f} MW")
        print(f"   Mean LMP: ${summary.mean_lmp_usd:.2f}/MWh")
        print(f"   EnKF RMSE: {summary.enkf_rmse:.2f} MW")
        print(f"   Mean Belief: {summary.mean_belief:.3f}")
        print(f"   Mean Action: {summary.mean_action_mw:.2f} MW")
        print(f"   Total Revenue: ${summary.total_revenue_usd:,.2f}")
        print(f"   Sharpe Ratio: {summary.sharpe_ratio:.3f}" if summary.sharpe_ratio else "   Sharpe Ratio: N/A")
    
    print("\n" + "=" * 60)
    print("🎉 All Done!")
    print("=" * 60)
    
    return results, run_dir


def main() -> None:
    """Main entry point with default parameters."""
    run_simulation(
        persist_results=True,
        generate_analysis=True,
        rl_timesteps=2_000,
        seed=42,  # For reproducibility
    )


if __name__ == "__main__":
    main()
