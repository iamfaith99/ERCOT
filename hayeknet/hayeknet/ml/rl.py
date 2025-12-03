"""Reinforcement learning components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover - optional heavy dependency
    PPO = None  # type: ignore
    make_vec_env = None  # type: ignore
    DummyVecEnv = None  # type: ignore


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

