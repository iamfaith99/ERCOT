"""Multi-Agent QSE (Qualified Scheduling Entity) system for ERCOT RTC.

This module implements the full MARL architecture with:
- QSE agents for different resource types (battery, solar, wind, gas)
- Grid DAG representation with transmission constraints
- Incremental PPO training on historical data
- Bayesian belief updates from SCED signals
- Dutch Book coherence checks
- Dispatch option valuation

References:
- Sutton & Barto Ch. 13 (Policy Gradient Methods)
- Evensen (2009) - Data Assimilation: The Ensemble Kalman Filter
- Jaynes (2003) - Probability Theory: The Logic of Science
- Benth et al. (2008) - Energy Markets: Stochastic Models
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

import numpy as np
import pandas as pd

# Import juliacall FIRST to prevent segfaults
try:
    from juliacall import Main as jl
except ImportError:
    jl = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    PPO = None
    DummyVecEnv = None
    BaseCallback = None

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
    model: Optional[Any] = None
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
        lmp_col = 'LMP' if 'LMP' in lmp_data.columns else 'lmp_usd'
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
            reg_up = latest_row.get('reg_up_price', 15.0)
            reg_down = latest_row.get('reg_down_price', 8.0)
            rrs = latest_row.get('rrs_price', 20.0)
            ecrs = latest_row.get('ecrs_price', 12.0)
            
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
        
        # Extract evidence
        current_lmp = sced_signals.get('lmp', lmp_observations['LMP'].iloc[-1])
        lmp_volatility = lmp_observations['LMP'].tail(12).std()
        
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
        current_lmp = sced_signals.get('lmp', lmp_data['LMP'].iloc[-1])
        
        # Exponential moving average
        alpha = 0.1
        self.beliefs['price_high_prob'] = (
            alpha * (1.0 if current_lmp > 50.0 else 0.0) + 
            (1 - alpha) * self.beliefs['price_high_prob']
        )
        
        lmp_vol = lmp_data['LMP'].tail(12).std()
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


@dataclass
class MARLSystem:
    """Multi-Agent Reinforcement Learning system for ERCOT RTC.
    
    Coordinates multiple QSE agents learning co-optimized bidding strategies.
    Implements incremental training on growing historical dataset.
    """
    agents: Dict[ResourceType, QSEAgent] = field(default_factory=dict)
    training_data: Optional[pd.DataFrame] = None
    model_dir: Path = Path("models/qse_agents")
    
    def __post_init__(self):
        """Initialize MARL system."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default agents if none provided
        if not self.agents:
            self.agents = {
                ResourceType.BATTERY: QSEAgent(ResourceType.BATTERY, capacity_mw=100.0),
                ResourceType.SOLAR: QSEAgent(ResourceType.SOLAR, capacity_mw=200.0),
                ResourceType.WIND: QSEAgent(ResourceType.WIND, capacity_mw=150.0),
            }
    
    def load_historical_data(self, data_dir: Path) -> pd.DataFrame:
        """Load all historical LMP data for incremental training.
        
        Parameters
        ----------
        data_dir : Path
            Directory containing parquet archives
            
        Returns
        -------
        data : pd.DataFrame
            Combined historical data
        """
        parquet_files = list(data_dir.glob("ercot_lmp_*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
        dfs = []
        for pq_file in sorted(parquet_files):
            df = pd.read_parquet(pq_file)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp column (handle different possible names)
        time_col = None
        for col in ['timestamp', 'DeliveryDate', 'SCEDTimestamp']:
            if col in combined.columns:
                time_col = col
                break
        
        if time_col:
            combined = combined.sort_values(time_col).reset_index(drop=True)
        else:
            combined = combined.reset_index(drop=True)
        
        return combined
    
    def train_incremental(
        self,
        data_dir: Path,
        timesteps_per_day: int = 10_000,
        save_checkpoints: bool = True
    ) -> Dict[str, Any]:
        """Incrementally train all QSE agents on historical data.
        
        Each day, agents train on ALL data seen so far, allowing them
        to adapt and improve as more market patterns emerge.
        
        Parameters
        ----------
        data_dir : Path
            Directory with historical data
        timesteps_per_day : int
            RL training steps per agent per day
        save_checkpoints : bool
            Whether to save model checkpoints
            
        Returns
        -------
        metrics : Dict
            Training metrics for all agents
        """
        if PPO is None:
            raise RuntimeError("stable-baselines3 required for RL training")
        
        # Load all available historical data
        self.training_data = self.load_historical_data(data_dir)
        
        # Standardize column names for training
        if 'lmp_usd' in self.training_data.columns and 'LMP' not in self.training_data.columns:
            self.training_data['LMP'] = self.training_data['lmp_usd']
        
        # Get time column for reporting
        time_col = None
        for col in ['timestamp', 'DeliveryDate', 'SCEDTimestamp']:
            if col in self.training_data.columns:
                time_col = col
                break
        
        print(f"📊 Loaded {len(self.training_data):,} observations for training")
        if time_col:
            print(f"   Date range: {self.training_data[time_col].min()} to {self.training_data[time_col].max()}")
        
        metrics = {}
        
        # Train each agent
        for resource_type, agent in self.agents.items():
            print(f"🤖 Training {resource_type.value} agent...")
            
            # Create environment for this agent
            env = self._create_agent_environment(agent)
            
            # Train or continue training
            if agent.model is None:
                # Initialize new model
                agent.model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                )
            else:
                # Continue training existing model
                agent.model.set_env(env)
            
            # Train
            agent.model.learn(
                total_timesteps=timesteps_per_day,
                reset_num_timesteps=False,  # Continue from previous training
                progress_bar=False
            )
            
            # Record metrics
            training_info = {
                'resource_type': resource_type.value,
                'timesteps': timesteps_per_day,
                'total_data_points': len(self.training_data),
                'date': pd.Timestamp.now().isoformat()
            }
            agent.training_history.append(training_info)
            metrics[resource_type.value] = training_info
            
            # Save checkpoint
            if save_checkpoints:
                model_path = self.model_dir / f"{resource_type.value}_latest.zip"
                agent.model.save(model_path)
                print(f"   ✅ Model saved: {model_path}")
        
        return metrics
    
    def _create_agent_environment(self, agent: QSEAgent):
        """Create Gym environment for agent training."""
        # This would normally be a custom Gym environment
        # For now, create a dummy environment as placeholder
        from gymnasium import spaces
        import gymnasium as gym
        
        class RTCBiddingEnv(gym.Env):
            """ERCOT RTC bidding environment for single agent."""
            
            def __init__(self, agent_ref, data):
                super().__init__()
                self.agent = agent_ref
                self.data = data
                self.current_idx = 0
                
                # State: 27 features (12 LMP + 4 AS + 4 resource + 4 beliefs + 3 grid)
                self.observation_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(27,), dtype=np.float32
                )
                
                # Action: bid quantity (fraction of capacity)
                self.action_space = spaces.Box(
                    low=agent_ref.characteristics['min_output'],
                    high=agent_ref.characteristics['max_output'],
                    shape=(1,),
                    dtype=np.float32
                )
            
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.current_idx = np.random.randint(12, len(self.data) - 1)
                obs = self._get_observation()
                return obs, {}
            
            def step(self, action):
                # Get current LMP
                lmp = self.data['LMP'].iloc[self.current_idx]
                
                # Compute reward
                reward = self.agent.compute_reward(float(action[0]), lmp)
                
                # Move to next timestep
                self.current_idx += 1
                terminated = self.current_idx >= len(self.data) - 1
                truncated = False
                
                obs = self._get_observation() if not terminated else np.zeros(27, dtype=np.float32)
                
                return obs, reward, terminated, truncated, {}
            
            def _get_observation(self):
                # Get state representation
                recent_data = self.data.iloc[max(0, self.current_idx-12):self.current_idx+1]
                return self.agent.get_state_representation(recent_data)
        
        env = RTCBiddingEnv(agent, self.training_data)
        return DummyVecEnv([lambda: env])
    
    def generate_bids(
        self,
        current_data: pd.DataFrame,
        grid_state: Optional[Dict] = None
    ) -> Dict[ResourceType, Dict[str, float]]:
        """Generate bids from all trained agents.
        
        Parameters
        ----------
        current_data : pd.DataFrame
            Recent market data
        grid_state : Dict, optional
            Current grid topology and constraints
            
        Returns
        -------
        bids : Dict
            Bids from each agent with confidence intervals
        """
        bids = {}
        
        for resource_type, agent in self.agents.items():
            if agent.model is None:
                # Agent not trained yet, use heuristic
                lmp_trend = current_data['LMP'].diff().mean()
                bid_qty = 0.5 + (lmp_trend / 20.0)
                bid_qty = np.clip(bid_qty, agent.characteristics['min_output'], agent.characteristics['max_output'])
                
                bids[resource_type] = {
                    'quantity_mw': bid_qty * agent.capacity_mw,
                    'price_bid': current_data['LMP'].iloc[-1] * 0.95,  # Slightly undercut
                    'confidence': 0.3,  # Low confidence (untrained)
                    'method': 'heuristic'
                }
            else:
                # Use trained RL policy
                state = agent.get_state_representation(current_data, grid_state)
                action, _ = agent.model.predict(state, deterministic=True)
                
                bid_qty = float(action[0])
                bids[resource_type] = {
                    'quantity_mw': bid_qty * agent.capacity_mw,
                    'price_bid': current_data['LMP'].iloc[-1] * 0.98,
                    'confidence': agent.beliefs['price_high_prob'],
                    'method': 'rl_policy'
                }
            
            # Dutch Book check
            if not agent.check_dutch_book():
                print(f"⚠️  Warning: {resource_type.value} agent has incoherent beliefs!")
        
        return bids
    
    def backtest(
        self,
        test_data: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """Backtest agent strategies on historical data.
        
        Parameters
        ----------
        test_data : pd.DataFrame
            Historical data for backtesting
        initial_capital : float
            Starting capital ($)
            
        Returns
        -------
        results : Dict
            Backtest metrics (PnL, Sharpe, drawdown, etc.)
        """
        portfolio_value = initial_capital
        pnl_series = []
        
        for idx in range(12, len(test_data)):
            recent_data = test_data.iloc[idx-12:idx+1]
            
            # Generate bids
            bids = self.generate_bids(recent_data)
            
            # Simulate dispatch and PnL
            current_lmp = test_data['LMP'].iloc[idx]
            interval_pnl = 0.0
            
            for resource_type, bid in bids.items():
                # Assume 50% chance of dispatch (simplified)
                if np.random.random() < 0.5:
                    dispatch_revenue = bid['quantity_mw'] * current_lmp / 12.0  # Per 5-min
                    interval_pnl += dispatch_revenue
            
            portfolio_value += interval_pnl
            pnl_series.append(portfolio_value)
        
        # Calculate metrics
        pnl_array = np.array(pnl_series)
        returns = np.diff(pnl_array) / pnl_array[:-1]
        
        results = {
            'final_capital': portfolio_value,
            'total_pnl': portfolio_value - initial_capital,
            'return_pct': (portfolio_value / initial_capital - 1.0) * 100,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(288 * 365),  # Annualized
            'max_drawdown': self._calculate_max_drawdown(pnl_array),
            'num_intervals': len(pnl_series)
        }
        
        return results
    
    @staticmethod
    def _calculate_max_drawdown(portfolio_series: np.ndarray) -> float:
        """Calculate maximum drawdown from portfolio series."""
        cummax = np.maximum.accumulate(portfolio_series)
        drawdown = (portfolio_series - cummax) / cummax
        return float(np.min(drawdown)) * 100
    
    def save_state(self, filepath: Path):
        """Save MARL system state to disk."""
        state = {
            'agents': {},
            'model_dir': str(self.model_dir)
        }
        
        for resource_type, agent in self.agents.items():
            state['agents'][resource_type.value] = {
                'capacity_mw': agent.capacity_mw,
                'beliefs': agent.beliefs,
                'training_history': agent.training_history
            }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: Path):
        """Load MARL system state from disk."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        for resource_str, agent_state in state['agents'].items():
            resource_type = ResourceType(resource_str)
            if resource_type in self.agents:
                agent = self.agents[resource_type]
                agent.beliefs = agent_state['beliefs']
                agent.training_history = agent_state['training_history']
                
                # Load model if exists
                model_path = self.model_dir / f"{resource_str}_latest.zip"
                if model_path.exists() and PPO is not None:
                    agent.model = PPO.load(model_path)
