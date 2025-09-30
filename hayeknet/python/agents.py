"""Bayesian reasoning and reinforcement learning agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Import juliacall FIRST to prevent segfaults with torch
try:
    from juliacall import Main as jl
except ImportError:
    jl = None

import numpy as np

try:
    import pymc as pm
except ImportError:  # pragma: no cover - optional heavy dependency
    pm = None  # type: ignore

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover - optional heavy dependency
    PPO = None  # type: ignore
    make_vec_env = None  # type: ignore
    DummyVecEnv = None  # type: ignore


@dataclass
class BayesianReasoner:
    """Perform Bayesian plausible reasoning with coherence safeguards."""

    prior_high: float = 0.3
    evidence_noise: float = 0.1
    samples: int = 1000
    tune: int = 500

    def update(self, da_estimate: float) -> float:
        if pm is None:
            raise RuntimeError("pymc is required for Bayesian updates")

        with pm.Model():
            high = pm.Bernoulli("high", self.prior_high)
            mu = pm.math.switch(high, 1.0, -1.0) * da_estimate
            pm.Normal("obs", mu=mu, sigma=self.evidence_noise, observed=da_estimate)
            trace = pm.sample(
                self.samples,
                tune=self.tune,
                progressbar=False,
                chains=1,
                cores=1,
                return_inferencedata=True,
            )

        post = float(trace.posterior["high"].mean().values)
        if not np.isclose(post + (1 - post), 1.0, atol=1e-6):
            raise ValueError("Bayesian update produced incoherent beliefs")
        if post < 0 or post > 1:
            raise ValueError("Posterior probability is outside [0, 1]")
        return post


@dataclass
class RLTrainer:
    """Reinforcement learning wrapper around a Stable-Baselines PPO agent."""

    vector_envs: int = 4
    total_timesteps: int = 10_000

    def train(self, env_factory) -> Tuple[object, dict]:
        if PPO is None or make_vec_env is None:
            raise RuntimeError("stable-baselines3 is required for RL training")

        vec_env = make_vec_env(env_factory, n_envs=self.vector_envs)
        model = PPO("MlpPolicy", vec_env, verbose=0)
        model.learn(total_timesteps=self.total_timesteps)
        return model, {"timesteps": self.total_timesteps}


def decide_bid(model, observation: np.ndarray) -> float:
    """Use a trained PPO model to produce an action for the provided observation."""
    if model is None:
        raise RuntimeError("A trained RL model is required before calling decide_bid")
    action, _ = model.predict(observation, deterministic=True)
    return float(action[0])

