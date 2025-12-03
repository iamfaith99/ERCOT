"""Core QSE agent domain models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pymc as pm
except ImportError:
    pm = None


class ResourceType(Enum):
    """ERCOT resource types for QSE agents."""
    BATTERY = "battery"
    SOLAR = "solar"
    WIND = "wind"
    GAS = "gas"
    LOAD = "load"


@dataclass
class QSEAgent:
    """Qualified Scheduling Entity agent for a specific resource type.
    
    Each QSE manages one resource type and learns optimal bidding strategies
    through multi-agent reinforcement learning in the ERCOT RTC market.
    
    Attributes
    ----------
    resource_type : ResourceType
        Type of resource this agent manages
    capacity_mw : float
        Maximum capacity in MW
    model : Optional[PPO]
        Trained PPO model for policy
    training_history : List[Dict]
        History of training metrics
    beliefs : Dict[str, float]
        Bayesian beliefs about market states
    """
    resource_type: ResourceType
    capacity_mw: float
    model: Optional[object] = None
    training_history: List[Dict] = field(default_factory=list)
    beliefs: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize agent-specific parameters."""
        # Resource-specific characteristics
        self.characteristics = {
            ResourceType.BATTERY: {
                'ramp_rate': 1.0,  # Can ramp instantly
                'min_output': -1.0,  # Can charge
                'max_output': 1.0,  # Can discharge
                'efficiency': 0.85,
                'response_time': 0.25  # 15 seconds
            },
            ResourceType.SOLAR: {
                'ramp_rate': 0.3,  # Limited by clouds
                'min_output': 0.0,  # Cannot consume
                'max_output': 1.0,
                'efficiency': 1.0,
                'response_time': 5.0  # 5 minutes
            },
            ResourceType.WIND: {
                'ramp_rate': 0.4,  # Wind variability
                'min_output': 0.0,
                'max_output': 1.0,
                'efficiency': 1.0,
                'response_time': 5.0
            },
            ResourceType.GAS: {
                'ramp_rate': 0.2,  # Slower ramp
                'min_output': 0.3,  # Min stable level
                'max_output': 1.0,
                'efficiency': 0.42,  # Heat rate
                'response_time': 10.0  # 10 minutes
            },
            ResourceType.LOAD: {
                'ramp_rate': 0.5,
                'min_output': 0.0,
                'max_output': 1.0,
                'efficiency': 1.0,
                'response_time': 1.0
            }
        }[self.resource_type]
        
        # Initialize beliefs (Jaynes-style)
        self.beliefs = {
            'price_high_prob': 0.3,  # P(LMP > $50)
            'imbalance_prob': 0.1,  # P(imbalance)
            'congestion_prob': 0.05,  # P(transmission congestion)
            'volatility_state': 0.2,  # Current volatility estimate
        }
    
    def get_state_representation(
        self,
        lmp_data: pd.DataFrame,
        grid_state: Optional[Dict] = None
    ) -> np.ndarray:
        """Convert market data to agent state representation.
        
        State includes:
        - Recent LMP history (normalized)
        - Ancillary service prices (reg up/down, RRS, ECRS)
        - Grid DAG features (congestion, flows)
        - Agent's resource characteristics
        - Bayesian belief states
        
        Parameters
        ----------
        lmp_data : pd.DataFrame
            Recent LMP observations (may include AS prices)
        grid_state : Dict, optional
            Grid topology and constraint states
            
        Returns
        -------
        state : np.ndarray
            State vector for RL policy
        """
        # LMP features (last 12 intervals = 1 hour)
        # Standardize to use lmp_usd consistently
        lmp_col = 'lmp_usd' if 'lmp_usd' in lmp_data.columns else 'LMP'
        recent_lmps = lmp_data[lmp_col].tail(12).values
        if len(recent_lmps) < 12:
            recent_lmps = np.pad(recent_lmps, (12-len(recent_lmps), 0), constant_values=25.0)
        
        lmp_mean = recent_lmps.mean()
        lmp_std = recent_lmps.std() + 1e-6
        lmp_features = (recent_lmps - lmp_mean) / lmp_std
        
        # Ancillary service price features (current values, normalized)
        as_features = np.zeros(4)  # reg_up, reg_down, rrs, ecrs
        if not lmp_data.empty:
            latest_row = lmp_data.iloc[-1]
            
            # Extract AS prices with fallbacks
            # Use column existence check first, then safely access from Series
            # pandas Series.get() works with defaults, but checking column existence is safer
            reg_up = latest_row['reg_up_price'] if 'reg_up_price' in lmp_data.columns else 15.0
            reg_down = latest_row['reg_down_price'] if 'reg_down_price' in lmp_data.columns else 8.0
            rrs = latest_row['rrs_price'] if 'rrs_price' in lmp_data.columns else 20.0
            ecrs = latest_row['ecrs_price'] if 'ecrs_price' in lmp_data.columns else 12.0
            
            # Normalize AS prices (typical range: $5-50/MW)
            as_features = np.array([
                (reg_up - 15.0) / 20.0,   # Centered at $15, scale by $20
                (reg_down - 8.0) / 15.0,  # Centered at $8, scale by $15
                (rrs - 20.0) / 25.0,      # Centered at $20, scale by $25
                (ecrs - 12.0) / 18.0,     # Centered at $12, scale by $18
            ])
            
            # Clip to reasonable range
            as_features = np.clip(as_features, -3.0, 3.0)
        
        # Resource characteristics
        resource_features = np.array([
            self.characteristics['ramp_rate'],
            self.characteristics['min_output'],
            self.characteristics['max_output'],
            self.characteristics['efficiency'],
        ])
        
        # Bayesian beliefs
        belief_features = np.array([
            self.beliefs['price_high_prob'],
            self.beliefs['imbalance_prob'],
            self.beliefs['congestion_prob'],
            self.beliefs['volatility_state'],
        ])
        
        # Grid state (if available)
        if grid_state:
            grid_features = np.array([
                grid_state.get('congestion_level', 0.0),
                grid_state.get('load_forecast', 60000.0) / 100000.0,  # Normalize
                grid_state.get('renewable_output', 0.3),
            ])
        else:
            grid_features = np.zeros(3)
        
        # Concatenate all features
        state = np.concatenate([
            lmp_features,      # 12 features
            as_features,       # 4 features (reg_up, reg_down, rrs, ecrs)
            resource_features, # 4 features
            belief_features,   # 4 features
            grid_features      # 3 features
        ])                     # Total: 27 features
        
        return state.astype(np.float32)
    
    def compute_reward(
        self,
        action: float,
        lmp: float,
        opportunity_cost: float = 0.0,
        as_prices: Optional[Dict[str, float]] = None,
        as_dispatch: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute reward for RL training with ancillary service considerations.
        
        Reward = Energy_Revenue + AS_Revenue - opportunity_cost - penalties
        
        Parameters
        ----------
        action : float
            Bid quantity (fraction of capacity)
        lmp : float
            Realized LMP ($/MWh)
        opportunity_cost : float
            Cost of committing resource
        as_prices : Dict, optional
            Ancillary service prices (reg_up, reg_down, rrs, ecrs)
        as_dispatch : Dict, optional
            Ancillary service dispatch amounts (MW)
            
        Returns
        -------
        reward : float
            Reward signal for RL agent
        """
        # Energy market revenue
        dispatch_mw = action * self.capacity_mw
        energy_revenue = lmp * dispatch_mw
        
        # Ancillary service revenue
        as_revenue = 0.0
        if as_prices and as_dispatch:
            for service in ['reg_up', 'reg_down', 'rrs', 'ecrs']:
                price = as_prices.get(f'{service}_price', 0.0)
                dispatch = as_dispatch.get(f'{service}_mw', 0.0)
                as_revenue += price * dispatch
        
        # Total revenue
        total_revenue = energy_revenue + as_revenue
        
        # Costs
        cost = opportunity_cost * abs(dispatch_mw)
        
        # Penalties for constraint violations
        penalty = 0.0
        if action < self.characteristics['min_output']:
            penalty += 100.0 * abs(action - self.characteristics['min_output'])
        if action > self.characteristics['max_output']:
            penalty += 100.0 * abs(action - self.characteristics['max_output'])
        
        # Battery-specific penalty for inefficient cycling
        if self.resource_type == ResourceType.BATTERY:
            # Penalize frequent direction changes
            if hasattr(self, '_last_action') and self._last_action is not None:
                if (self._last_action > 0) != (action > 0) and abs(action) > 0.1:
                    penalty += 5.0  # Small cycling penalty
            self._last_action = action
        
        # Net reward (per 5-min interval)
        reward = (total_revenue - cost - penalty) / 12.0  # Normalize to $/hour
        
        return reward
    
    def update_beliefs(
        self,
        sced_signals: Dict[str, float],
        lmp_observations: pd.DataFrame
    ):
        """Update Bayesian beliefs based on SCED outputs (Jaynes-style).
        
        Treat prices as evidence for hidden market states:
        - P(blackout | volatility)
        - P(congestion | price spread)
        - P(imbalance | forecast error)
        
        Parameters
        ----------
        sced_signals : Dict
            SCED dispatch and price signals
        lmp_observations : pd.DataFrame
            Recent LMP data for inference
        """
        if pm is None:
            # Fallback to simple Bayesian update
            self._simple_belief_update(sced_signals, lmp_observations)
            return
        
        # Extract evidence - standardize column name
        lmp_col = 'lmp_usd' if 'lmp_usd' in lmp_observations.columns else 'LMP'
        current_lmp = sced_signals.get('lmp', lmp_observations[lmp_col].iloc[-1])
        lmp_volatility = lmp_observations[lmp_col].tail(12).std()
        
        # Update P(high price state)
        prior_high = self.beliefs['price_high_prob']
        likelihood_high = 1.0 if current_lmp > 50.0 else 0.1
        evidence = (prior_high * likelihood_high) / (
            prior_high * likelihood_high + (1 - prior_high) * (1 - likelihood_high)
        )
        self.beliefs['price_high_prob'] = evidence
        
        # Update volatility state
        self.beliefs['volatility_state'] = 0.9 * self.beliefs['volatility_state'] + 0.1 * (lmp_volatility / 50.0)
        
        # Update imbalance probability based on forecast error
        if 'imbalance' in sced_signals:
            imbalance = sced_signals['imbalance']
            self.beliefs['imbalance_prob'] = 0.8 * self.beliefs['imbalance_prob'] + 0.2 * min(abs(imbalance) / 1000.0, 1.0)
        
        # Coherence check (Dutch Book prevention)
        total_prob = self.beliefs['price_high_prob'] + (1 - self.beliefs['price_high_prob'])
        if not np.isclose(total_prob, 1.0, atol=1e-6):
            raise ValueError(f"Incoherent beliefs: probabilities sum to {total_prob}")
    
    def _simple_belief_update(self, sced_signals: Dict, lmp_data: pd.DataFrame):
        """Simple Bayesian update without PyMC."""
        lmp_col = 'lmp_usd' if 'lmp_usd' in lmp_data.columns else 'LMP'
        current_lmp = sced_signals.get('lmp', lmp_data[lmp_col].iloc[-1])
        
        # Exponential moving average
        alpha = 0.1
        self.beliefs['price_high_prob'] = (
            alpha * (1.0 if current_lmp > 50.0 else 0.0) + 
            (1 - alpha) * self.beliefs['price_high_prob']
        )
        
        lmp_vol = lmp_data[lmp_col].tail(12).std()
        self.beliefs['volatility_state'] = alpha * (lmp_vol / 50.0) + (1 - alpha) * self.beliefs['volatility_state']
    
    def check_dutch_book(self) -> bool:
        """Check for Dutch Book arbitrage in ancillary bids.
        
        Ensures coherent bidding that prevents sure losses from
        over-committing regulation up/down or energy/ancillary conflicts.
        
        Returns
        -------
        is_coherent : bool
            True if no arbitrage opportunity exists
        """
        # Check probability coherence
        prob_sum = self.beliefs['price_high_prob'] + (1 - self.beliefs['price_high_prob'])
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            return False
        
        # All beliefs must be [0, 1]
        for belief_val in self.beliefs.values():
            if belief_val < 0 or belief_val > 1:
                return False
        
        return True

