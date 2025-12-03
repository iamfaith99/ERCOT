"""Simplified RTC trading environment with RTC+B support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from hayeknet.core.battery import BatterySimulator, BatterySpecs
from hayeknet.core.market import MarketDesign, RTCPlusBSimulator, SCEDSimulator, BidDecision


@dataclass
class RTCEnvConfig:
    max_bid_mw: float = 150.0
    imbalance_penalty: float = 0.1
    market_design: MarketDesign = MarketDesign.RTC_PLUS_B
    battery_specs: Optional[BatterySpecs] = None


class RTCEnv(gym.Env):
    """
    Gym-compatible environment for co-optimized bidding experiments.
    
    Enhanced with RTC+B features:
    - ASDC price formation
    - SOC tracking and constraints
    - Co-optimized energy + AS rewards
    - Unified bid curve support
    - Market design selection (SCED vs RTC+B)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        data_frame,
        beliefs: np.ndarray,
        config: RTCEnvConfig | None = None,
        battery: Optional[BatterySimulator] = None,
    ):
        super().__init__()
        self.data = data_frame.reset_index(drop=True)
        self.beliefs = beliefs
        self.config = config or RTCEnvConfig()
        
        # Initialize battery if not provided
        if battery is None:
            battery_specs = self.config.battery_specs or BatterySpecs(
                max_charge_mw=100.0,
                max_discharge_mw=100.0,
                capacity_mwh=400.0,
            )
            battery = BatterySimulator(battery_specs)
        self.battery = battery
        
        # Initialize market simulator based on design
        if self.config.market_design == MarketDesign.RTC_PLUS_B:
            self.market_sim = RTCPlusBSimulator(self.battery, asdc_enabled=True)
        else:
            self.market_sim = SCEDSimulator(self.battery)
        
        # Enhanced observation space: [load, LMP, belief, SOC, reg_up_price, rrs_price, 
        #                              reg_down_price, ecrs_price, market_design, asdc_scarcity]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # Action space: bid quantity (can be negative for charge, positive for discharge)
        # For RTC+B, supports unified bid curve from -max_charge to +max_discharge
        max_action = max(self.battery.specs.max_charge_mw, self.battery.specs.max_discharge_mw)
        self.action_space = spaces.Box(
            low=-max_action, high=max_action, shape=(1,), dtype=np.float32
        )
        
        self.current_step = 0

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment with RTC+B market clearing."""
        # Clip action to valid range
        bid_mw = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        
        row = self.data.iloc[self.current_step]
        belief = float(self.beliefs[self.current_step])
        
        # Extract market data
        lmp = float(row.get("lmp_usd", row.get("LMP", 0.0)))
        load = float(row.get("net_load_mw", row.get("total_load_mw", 0.0)))
        
        # Get AS prices (with defaults if not available)
        reg_up_price = float(row.get("reg_up_price", lmp * 0.5))
        reg_down_price = float(row.get("reg_down_price", lmp * 0.3))
        rrs_price = float(row.get("rrs_price", lmp * 0.4))
        ecrs_price = float(row.get("ecrs_price", lmp * 0.35))
        
        # Create market data series for simulator
        market_data = pd.Series({
            "lmp_usd": lmp,
            "net_load_mw": load,
            "reg_up_price": reg_up_price,
            "reg_down_price": reg_down_price,
            "rrs_price": rrs_price,
            "ecrs_price": ecrs_price,
        })
        
        # Create bid decision
        # For RTC+B: bid can be negative (charge) or positive (discharge)
        # For SCED: simplified to single direction
        if self.config.market_design == MarketDesign.RTC_PLUS_B:
            # RTC+B: unified bid curve - bid can be charge or discharge
            energy_bid = bid_mw
            energy_price = lmp * (1.1 if bid_mw < 0 else 0.9)  # Charge pays more, discharge accepts less
        else:
            # SCED: only positive bids (discharge)
            energy_bid = max(0.0, bid_mw)
            energy_price = lmp * 0.9
        
        bid = BidDecision(
            energy_bid_mw=energy_bid,
            energy_price_offer=energy_price,
        )
        
        # Clear market using appropriate simulator
        outcome = self.market_sim.clear_market(bid, market_data)
        
        # Execute battery operation
        self.battery.step(
            outcome.energy_cleared_mw,
            reg_up_mw=outcome.reg_up_cleared_mw,
            reg_down_mw=outcome.reg_down_cleared_mw,
            rrs_mw=outcome.rrs_cleared_mw,
            ecrs_mw=outcome.ecrs_cleared_mw,
        )
        
        # Calculate reward: co-optimized energy + AS revenue
        energy_revenue = outcome.energy_revenue
        as_revenue = (
            outcome.reg_up_revenue +
            outcome.reg_down_revenue +
            outcome.rrs_revenue +
            outcome.ecrs_revenue
        )
        
        # Penalties for constraint violations
        soc_penalty = 0.0
        if self.battery.state.soc < self.battery.specs.min_soc + 0.05:
            soc_penalty = -100.0  # Penalty for low SOC
        elif self.battery.state.soc > self.battery.specs.max_soc - 0.05:
            soc_penalty = -50.0  # Smaller penalty for high SOC
        
        # Imbalance penalty (if bid not fully cleared)
        imbalance = abs(bid_mw) - abs(outcome.energy_cleared_mw)
        imbalance_penalty = -self.config.imbalance_penalty * imbalance**2
        
        # Total reward
        reward = energy_revenue + as_revenue + soc_penalty + imbalance_penalty
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= len(self.data)
        truncated = False
        
        obs = self._build_observation()
        info = {
            "cleared": outcome.energy_cleared_mw,
            "lmp": lmp,
            "load": load,
            "belief": belief,
            "soc": self.battery.state.soc,
            "energy_revenue": energy_revenue,
            "as_revenue": as_revenue,
            "total_revenue": energy_revenue + as_revenue,
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset battery to initial state
        self.battery.reset()
        
        # Reset market simulator
        if self.config.market_design == MarketDesign.RTC_PLUS_B:
            self.market_sim = RTCPlusBSimulator(self.battery, asdc_enabled=True)
        else:
            self.market_sim = SCEDSimulator(self.battery)
        
        observation = self._build_observation()
        info = {"soc": self.battery.state.soc}
        return observation, info

    def _build_observation(self) -> np.ndarray:
        """Build enhanced observation with RTC+B features."""
        if self.current_step >= len(self.data):
            return np.zeros(10, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        belief = float(self.beliefs[self.current_step])
        
        # Extract market data
        load = float(row.get("net_load_mw", row.get("total_load_mw", 0.0)))
        lmp = float(row.get("lmp_usd", row.get("LMP", 0.0)))
        
        # Get AS prices
        reg_up_price = float(row.get("reg_up_price", lmp * 0.5))
        reg_down_price = float(row.get("reg_down_price", lmp * 0.3))
        rrs_price = float(row.get("rrs_price", lmp * 0.4))
        ecrs_price = float(row.get("ecrs_price", lmp * 0.35))
        
        # Get SOC
        soc = self.battery.state.soc
        
        # Market design indicator (1.0 for RTC+B, 0.0 for SCED)
        market_design_indicator = 1.0 if self.config.market_design == MarketDesign.RTC_PLUS_B else 0.0
        
        # ASDC scarcity indicator (simplified: based on load)
        # Higher load = higher scarcity
        asdc_scarcity = min(1.0, max(0.0, (load - 50000) / 20000)) if load > 0 else 0.0
        
        # Build observation: [load, LMP, belief, SOC, reg_up_price, rrs_price, 
        #                     reg_down_price, ecrs_price, market_design, asdc_scarcity]
        obs = np.array([
            load / 1000.0,  # Normalize load (divide by 1000)
            lmp / 100.0,    # Normalize LMP (divide by 100)
            belief,
            soc,            # SOC is already 0-1
            reg_up_price / 100.0,   # Normalize AS prices
            rrs_price / 100.0,
            reg_down_price / 100.0,
            ecrs_price / 100.0,
            market_design_indicator,
            asdc_scarcity,
        ], dtype=np.float32)
        
        return obs

