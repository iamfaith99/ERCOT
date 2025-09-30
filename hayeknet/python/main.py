"""Entry point for running a small HayekNet main simulation loop."""
from __future__ import annotations

from pathlib import Path

# Import juliacall FIRST to prevent segfaults with torch
try:
    from juliacall import Main as jl
except ImportError:
    jl = None

import numpy as np

from envs.rtc_env import RTCEnv, RTCEnvConfig
from python.agents import BayesianReasoner, RLTrainer, decide_bid
from python.data import ERCOTDataClient, build_observation_operator
from python.utils import init_julia, price_option, run_enkf, validate_constraints


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    init_julia(repo_root)

    data_client = ERCOTDataClient()
    df, obs_matrix = data_client.fetch_rtc_like()

    ensemble_size = 100
    state_dim = obs_matrix.shape[0]

    H = build_observation_operator(obs_matrix)
    R = np.eye(state_dim) * 15.0

    prior_state = np.repeat(obs_matrix[:, :1], ensemble_size, axis=1)
    perturbations = np.random.normal(0, 50, size=prior_state.shape)
    prior_ensemble = prior_state + perturbations

    obs_ensemble = np.repeat(obs_matrix[:, :1], ensemble_size, axis=1)
    mean_state, analysis = run_enkf(prior_ensemble, obs_ensemble, H, R, inflation=1.05)

    reasoner = BayesianReasoner(prior_high=0.35)
    belief = reasoner.update(float(mean_state[0]))
    beliefs = np.full(len(df), belief, dtype=float)

    config = RTCEnvConfig()
    env = RTCEnv(df, beliefs, config)

    rl_info = {}
    try:
        trainer = RLTrainer(vector_envs=2, total_timesteps=2_000)
        model, rl_info = trainer.train(lambda: RTCEnv(df, beliefs, config))
        # Create a test environment for prediction
        test_env = RTCEnv(df, beliefs, config)
        observation, _ = test_env.reset()  # Gymnasium returns (obs, info)
        bid = decide_bid(model, observation)
    except RuntimeError as exc:
        bid = float(config.max_bid_mw / 2)
        rl_info = {"warning": str(exc)}

    edges = [[1, 2], [2, 3], [3, 4]]  # Julia expects Vector{Vector{Int}}
    constraints_ok = validate_constraints(edges, 1, 4, capacity=200.0, bid=bid)

    hedge = price_option(
        float(df["lmp_usd"].mean()),
        strike=float(df["lmp_usd"].quantile(0.9)),
        rate=0.05,
        volatility=0.2,
        maturity=0.25,
        steps=32,
        trajectories=256,
    )

    print("HayekNet bootstrap complete")
    print(f"DA mean state: {mean_state}")
    print(f"Bayesian belief (high demand): {belief:.3f}")
    print(f"RL bid suggestion: {bid:.2f} MW")
    print(f"Constraints satisfied: {constraints_ok}")
    print(f"Option hedge price: ${hedge:.2f}")
    if rl_info:
        print(f"RL metadata: {rl_info}")


if __name__ == "__main__":
    main()

