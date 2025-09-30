"""Simplified RTC trading environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RTCEnvConfig:
    max_bid_mw: float = 150.0
    imbalance_penalty: float = 0.1


class RTCEnv(gym.Env):
    """Gym-compatible environment for co-optimized bidding experiments."""

    metadata = {"render.modes": []}

    def __init__(self, data_frame, beliefs: np.ndarray, config: RTCEnvConfig | None = None):
        super().__init__()
        self.data = data_frame.reset_index(drop=True)
        self.beliefs = beliefs
        self.config = config or RTCEnvConfig()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=self.config.max_bid_mw, shape=(1,), dtype=np.float32)
        self.current_step = 0

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        bid = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        row = self.data.iloc[self.current_step]
        belief = float(self.beliefs[self.current_step])

        lmp = float(row.get("lmp_usd", 0.0))
        load = float(row.get("net_load_mw", 0.0))

        cleared = min(bid, load)
        imbalance = bid - cleared
        reward = cleared * lmp - self.config.imbalance_penalty * imbalance**2

        self.current_step += 1
        terminated = self.current_step >= len(self.data)
        truncated = False  # No time limits in this environment
        obs = self._build_observation()
        info = {"cleared": cleared, "lmp": lmp, "load": load, "belief": belief}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        observation = self._build_observation()
        info = {}
        return observation, info

    def _build_observation(self) -> np.ndarray:
        if self.current_step >= len(self.data):
            return np.zeros(3, dtype=np.float32)
        row = self.data.iloc[self.current_step]
        belief = float(self.beliefs[self.current_step])
        return np.array([row.get("net_load_mw", 0.0), row.get("lmp_usd", 0.0), belief], dtype=np.float32)

