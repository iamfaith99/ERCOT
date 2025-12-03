"""Multi-Agent Reinforcement Learning system for QSE agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    PPO = None
    DummyVecEnv = None

from hayeknet.core.agents import ResourceType, QSEAgent
from hayeknet.core.market import MarketDesign


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
        
        # Standardize column names for training - use lmp_usd consistently
        if 'lmp_usd' in self.training_data.columns and 'LMP' not in self.training_data.columns:
            self.training_data['LMP'] = self.training_data['lmp_usd']
        elif 'LMP' in self.training_data.columns and 'lmp_usd' not in self.training_data.columns:
            self.training_data['lmp_usd'] = self.training_data['LMP']
        
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
                # Initialize new model with ERCOT-optimized hyperparameters
                agent.model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    learning_rate=1e-4,
                    n_steps=1024,
                    batch_size=128,
                    n_epochs=5,
                    gamma=0.98,
                    ent_coef=0.01,
                    gae_lambda=0.95,
                    clip_range=0.2,
                )
            else:
                # Continue training existing model
                agent.model.set_env(env)
            
            # Train
            agent.model.learn(
                total_timesteps=timesteps_per_day,
                reset_num_timesteps=False,
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
        """Create Gym environment for agent training with RTC+B support."""
        from gymnasium import spaces
        import gymnasium as gym
        from hayeknet.core.market import MarketDesign, RTCPlusBSimulator, SCEDSimulator, BidDecision
        from hayeknet.core.battery import BatterySimulator, BatterySpecs
        
        class RTCBiddingEnv(gym.Env):
            """ERCOT RTC bidding environment for single agent with RTC+B features."""
            
            def __init__(self, agent_ref, data, market_design: MarketDesign = MarketDesign.RTC_PLUS_B):
                super().__init__()
                self.agent = agent_ref
                self.data = data
                self.current_idx = 0
                self.market_design = market_design
                
                # Enhanced state: 27 base + 5 RTC+B features = 32 features for battery
                # Base: 12 LMP + 4 AS + 4 resource + 4 beliefs + 3 grid = 27
                # RTC+B: SOC (if battery), ASDC scarcity, market_design, unified_curve_indicator, coopt_value
                # For non-battery: 27 base + 3 RTC+B features = 30 features
                # RTC+B: ASDC scarcity, market_design, unified_curve_indicator (no SOC or coopt)
                if agent_ref.resource_type.value == 'battery':
                    state_size = 32
                else:
                    state_size = 30
                self.observation_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(state_size,), dtype=np.float32
                )
                
                # Action: bid quantity (fraction of capacity)
                # For RTC+B battery: can be negative (charge) or positive (discharge)
                if agent_ref.resource_type.value == 'battery' and market_design == MarketDesign.RTC_PLUS_B:
                    # Unified bid curve: -1.0 (max charge) to +1.0 (max discharge)
                    self.action_space = spaces.Box(
                        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                    )
                else:
                    # Traditional: 0 to max_output
                    self.action_space = spaces.Box(
                        low=agent_ref.characteristics['min_output'],
                        high=agent_ref.characteristics['max_output'],
                        shape=(1,),
                        dtype=np.float32
                    )
                
                # Initialize battery simulator for battery agents
                if agent_ref.resource_type.value == 'battery':
                    battery_specs = BatterySpecs(
                        max_charge_mw=agent_ref.capacity_mw,
                        max_discharge_mw=agent_ref.capacity_mw,
                        capacity_mwh=agent_ref.capacity_mw * 4.0,  # 4-hour duration
                    )
                    self.battery = BatterySimulator(battery_specs)
                    
                    # Initialize market simulator
                    if market_design == MarketDesign.RTC_PLUS_B:
                        self.market_sim = RTCPlusBSimulator(self.battery, asdc_enabled=True)
                    else:
                        self.market_sim = SCEDSimulator(self.battery)
                else:
                    self.battery = None
                    self.market_sim = None
            
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.current_idx = np.random.randint(12, len(self.data) - 1)
                if self.battery:
                    self.battery.reset()
                obs = self._get_observation()
                return obs, {}
            
            def step(self, action):
                """Execute step with RTC+B market clearing."""
                action_val = float(action[0])
                
                # Get current market data
                lmp_col = 'lmp_usd' if 'lmp_usd' in self.data.columns else 'LMP'
                row = self.data.iloc[self.current_idx]
                lmp = float(row.get(lmp_col, 0.0))
                
                # Get AS prices
                reg_up_price = float(row.get('reg_up_price', lmp * 0.5))
                reg_down_price = float(row.get('reg_down_price', lmp * 0.3))
                rrs_price = float(row.get('rrs_price', lmp * 0.4))
                ecrs_price = float(row.get('ecrs_price', lmp * 0.35))
                
                # Create market data series
                market_data = pd.Series({
                    lmp_col: lmp,
                    'lmp_usd': lmp,
                    'reg_up_price': reg_up_price,
                    'reg_down_price': reg_down_price,
                    'rrs_price': rrs_price,
                    'ecrs_price': ecrs_price,
                    'net_load_mw': row.get('net_load_mw', 60000.0),
                })
                
                # Compute reward based on market design
                if self.battery and self.market_sim:
                    # Battery agent with RTC+B market clearing
                    # Convert action to MW (action is fraction for battery)
                    if self.market_design == MarketDesign.RTC_PLUS_B:
                        # Unified bid curve: action can be negative (charge) or positive (discharge)
                        bid_mw = action_val * self.agent.capacity_mw
                    else:
                        # SCED: only positive bids
                        bid_mw = max(0.0, action_val) * self.agent.capacity_mw
                    
                    # Create bid decision
                    bid = BidDecision(
                        energy_bid_mw=bid_mw,
                        energy_price_offer=lmp * (1.1 if bid_mw < 0 else 0.9),
                        reg_up_bid_mw=0.1 * self.agent.capacity_mw if action_val > 0 else 0.0,
                        reg_down_bid_mw=0.1 * self.agent.capacity_mw if action_val < 0 else 0.0,
                        reg_up_price=reg_up_price * 0.95,
                        reg_down_price=reg_down_price * 0.95,
                        rrs_price=rrs_price * 0.95,
                    )
                    
                    # Clear market
                    outcome = self.market_sim.clear_market(bid, market_data)
                    
                    # Execute battery operation
                    self.battery.step(
                        outcome.energy_cleared_mw,
                        reg_up_mw=outcome.reg_up_cleared_mw,
                        reg_down_mw=outcome.reg_down_cleared_mw,
                        rrs_mw=outcome.rrs_cleared_mw,
                        ecrs_mw=outcome.ecrs_cleared_mw,
                    )
                    
                    # Reward: co-optimized energy + AS revenue
                    reward = outcome.total_revenue / 1000.0  # Normalize reward
                else:
                    # Non-battery agent: use traditional reward
                    reward = self.agent.compute_reward(action_val, lmp)
                
                # Move to next timestep
                self.current_idx += 1
                terminated = self.current_idx >= len(self.data) - 1
                truncated = False
                
                obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape[0], dtype=np.float32)
                
                info = {
                    'lmp': lmp,
                    'action': action_val,
                    'reward': reward,
                }
                if self.battery:
                    info['soc'] = self.battery.state.soc
                
                return obs, reward, terminated, truncated, info
            
            def _get_observation(self):
                """Get enhanced state representation with RTC+B features."""
                recent_data = self.data.iloc[max(0, self.current_idx-12):self.current_idx+1]
                base_state = self.agent.get_state_representation(recent_data)
                
                # Add RTC+B features
                rtc_features = []
                
                # Market design indicator
                rtc_features.append(1.0 if self.market_design == MarketDesign.RTC_PLUS_B else 0.0)
                
                # ASDC scarcity indicator (based on load)
                if not recent_data.empty:
                    latest_row = recent_data.iloc[-1]
                    # Safely access column - check existence first
                    load = latest_row['net_load_mw'] if 'net_load_mw' in recent_data.columns else 60000.0
                    asdc_scarcity = min(1.0, max(0.0, (load - 50000) / 20000))
                else:
                    asdc_scarcity = 0.0
                rtc_features.append(asdc_scarcity)
                
                # Battery-specific features (only for battery agents)
                if self.battery and self.agent.resource_type.value == 'battery':
                    rtc_features.append(self.battery.state.soc)  # SOC
                    # Co-optimization value indicator (simplified)
                    coopt_value = 0.5  # Placeholder - could be calculated from AS prices
                    rtc_features.append(coopt_value)
                    # Unified curve indicator (1.0 if RTC+B battery, 0.0 otherwise)
                    unified_indicator = 1.0 if self.market_design == MarketDesign.RTC_PLUS_B else 0.0
                    rtc_features.append(unified_indicator)
                else:
                    # For non-battery: only unified curve indicator (no SOC or coopt)
                    unified_indicator = 0.0  # Non-battery agents don't use unified curves
                    rtc_features.append(unified_indicator)
                
                # Concatenate base state with RTC+B features
                enhanced_state = np.concatenate([base_state, np.array(rtc_features, dtype=np.float32)])
                
                # Ensure state size matches observation space
                expected_size = self.observation_space.shape[0]
                if len(enhanced_state) != expected_size:
                    # Pad or truncate to match expected size
                    if len(enhanced_state) < expected_size:
                        enhanced_state = np.pad(enhanced_state, (0, expected_size - len(enhanced_state)), 'constant')
                    else:
                        enhanced_state = enhanced_state[:expected_size]
                
                return enhanced_state
        
        env = RTCBiddingEnv(agent, self.training_data, market_design=MarketDesign.RTC_PLUS_B)
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
        
        # Standardize column name
        lmp_col = 'lmp_usd' if 'lmp_usd' in current_data.columns else 'LMP'
        if lmp_col not in current_data.columns:
            raise ValueError(f"LMP column not found in data. Available columns: {list(current_data.columns)}")
        
        for resource_type, agent in self.agents.items():
            if agent.model is None:
                # Agent not trained yet, use heuristic
                lmp_trend = current_data[lmp_col].diff().mean()
                bid_qty = 0.5 + (lmp_trend / 20.0)
                bid_qty = np.clip(bid_qty, agent.characteristics['min_output'], agent.characteristics['max_output'])
                
                bids[resource_type] = {
                    'quantity_mw': bid_qty * agent.capacity_mw,
                    'price_bid': current_data[lmp_col].iloc[-1] * 0.95,
                    'confidence': 0.3,
                    'method': 'heuristic'
                }
            else:
                # Use trained RL policy
                state = agent.get_state_representation(current_data, grid_state)
                action, _ = agent.model.predict(state, deterministic=True)
                
                bid_qty = float(action[0])
                bids[resource_type] = {
                    'quantity_mw': bid_qty * agent.capacity_mw,
                    'price_bid': current_data[lmp_col].iloc[-1] * 0.98,
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
        
        # Standardize column name
        lmp_col = 'lmp_usd' if 'lmp_usd' in test_data.columns else 'LMP'
        
        for idx in range(12, len(test_data)):
            recent_data = test_data.iloc[idx-12:idx+1]
            
            # Generate bids
            bids = self.generate_bids(recent_data)
            
            # Simulate dispatch and PnL
            current_lmp = test_data[lmp_col].iloc[idx]
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
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(288 * 365),
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
        """Save MARL system state to disk with training timestamp."""
        from datetime import datetime
        
        state = {
            'agents': {},
            'model_dir': str(self.model_dir),
            'last_training_date': datetime.now().isoformat(),
            'version': '1.1'
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
        """Load MARL system state from disk with backward compatibility."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Handle backward compatibility for training timestamp
        if 'last_training_date' not in state:
            state['last_training_date'] = '2020-01-01T00:00:00'
        
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

