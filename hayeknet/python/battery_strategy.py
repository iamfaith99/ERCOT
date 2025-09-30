"""Battery bidding strategies for ERCOT markets."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from python.battery_model import BatterySimulator
from python.market_simulator import BidDecision, MarketDesign


class BiddingStrategy(ABC):
    """Abstract base class for battery bidding strategies."""
    
    @abstractmethod
    def generate_bid(
        self,
        battery: BatterySimulator,
        market_data: pd.Series,
        market_design: MarketDesign,
    ) -> BidDecision:
        """
        Generate bid decision for current interval.
        
        Parameters
        ----------
        battery : BatterySimulator
            Current battery state
        market_data : pd.Series
            Market data for this interval
        market_design : MarketDesign
            Which market design (SCED or RTC+B)
            
        Returns
        -------
        BidDecision
            Bid to submit
        """
        pass


class SimpleArbitrageStrategy(BiddingStrategy):
    """
    Simple price arbitrage strategy.
    
    Logic:
    - Charge when prices are low
    - Discharge when prices are high
    - Threshold-based decisions
    - Participate in AS if more profitable than energy arbitrage
    """
    
    def __init__(
        self,
        charge_threshold_percentile: float = 0.3,
        discharge_threshold_percentile: float = 0.7,
        as_premium_threshold: float = 1.5,
    ):
        """
        Initialize arbitrage strategy.
        
        Parameters
        ----------
        charge_threshold_percentile : float
            Charge when price below this percentile (default: 30th)
        discharge_threshold_percentile : float
            Discharge when price above this percentile (default: 70th)
        as_premium_threshold : float
            Minimum AS/energy price ratio to prefer AS (default: 1.5x)
        """
        self.charge_threshold_pct = charge_threshold_percentile
        self.discharge_threshold_pct = discharge_threshold_percentile
        self.as_premium_threshold = as_premium_threshold
        
        # Running statistics for adaptive thresholds
        self.price_history: list[float] = []
        self.max_history = 288  # 24 hours at 5-min intervals
    
    def generate_bid(
        self,
        battery: BatterySimulator,
        market_data: pd.Series,
        market_design: MarketDesign,
    ) -> BidDecision:
        """Generate arbitrage-focused bid."""
        lmp = float(market_data.get("lmp_usd", 0))
        
        # Update price history
        self.price_history.append(lmp)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        
        # Compute adaptive thresholds
        if len(self.price_history) >= 12:  # Need at least 1 hour of data
            price_low = np.percentile(self.price_history, self.charge_threshold_pct * 100)
            price_high = np.percentile(self.price_history, self.discharge_threshold_pct * 100)
        else:
            # Use static thresholds until enough data
            price_low = 30.0
            price_high = 50.0
        
        # Get ancillary service prices
        reg_up_price = float(market_data.get("reg_up_price", 0))
        reg_down_price = float(market_data.get("reg_down_price", 0))
        rrs_price = float(market_data.get("rrs_price", 0))
        ecrs_price = float(market_data.get("ecrs_price", 0))
        
        # Calculate AS premium relative to energy
        as_premium = (reg_up_price + rrs_price) / (lmp + 1e-6)
        
        # Decision logic
        bid = BidDecision(energy_bid_mw=0.0, energy_price_offer=lmp)
        
        if market_design == MarketDesign.SCED:
            # SCED: Choose between energy arbitrage OR AS
            if as_premium > self.as_premium_threshold:
                # AS more attractive
                bid = self._bid_ancillary_services_sced(battery, market_data)
            else:
                # Energy arbitrage
                bid = self._bid_energy_arbitrage(battery, lmp, price_low, price_high)
                
        else:  # RTC+B
            # RTC+B: Can do both energy and AS
            energy_bid = self._bid_energy_arbitrage(battery, lmp, price_low, price_high)
            as_bid = self._bid_ancillary_services_rtcb(battery, market_data, energy_bid)
            
            # Combine bids
            bid = BidDecision(
                energy_bid_mw=energy_bid.energy_bid_mw,
                energy_price_offer=energy_bid.energy_price_offer,
                reg_up_bid_mw=as_bid.reg_up_bid_mw,
                reg_down_bid_mw=as_bid.reg_down_bid_mw,
                rrs_bid_mw=as_bid.rrs_bid_mw,
                ecrs_bid_mw=as_bid.ecrs_bid_mw,
                reg_up_price=as_bid.reg_up_price,
                reg_down_price=as_bid.reg_down_price,
                rrs_price=as_bid.rrs_price,
                ecrs_price=as_bid.ecrs_price,
            )
        
        return bid
    
    def _bid_energy_arbitrage(
        self,
        battery: BatterySimulator,
        lmp: float,
        price_low: float,
        price_high: float,
    ) -> BidDecision:
        """Generate energy arbitrage bid."""
        soc = battery.state.soc
        
        # Charge when price is low and SOC < 80%
        if lmp < price_low and soc < 0.8:
            charge_capacity = battery.get_available_power(for_discharge=False)
            return BidDecision(
                energy_bid_mw=-charge_capacity,  # Negative = charge
                energy_price_offer=price_low * 1.1,  # Willing to pay up to 110% of threshold
            )
        
        # Discharge when price is high and SOC > 20%
        elif lmp > price_high and soc > 0.2:
            discharge_capacity = battery.get_available_power(for_discharge=True)
            return BidDecision(
                energy_bid_mw=discharge_capacity,  # Positive = discharge
                energy_price_offer=price_high * 0.9,  # Offer at 90% of threshold
            )
        
        # Hold position
        else:
            return BidDecision(energy_bid_mw=0.0, energy_price_offer=lmp)
    
    def _bid_ancillary_services_sced(
        self,
        battery: BatterySimulator,
        market_data: pd.Series,
    ) -> BidDecision:
        """Generate AS-focused bid for SCED (mutually exclusive with energy)."""
        reg_up_price = float(market_data.get("reg_up_price", 0))
        rrs_price = float(market_data.get("rrs_price", 0))
        
        # Offer all available capacity to AS
        available = battery.get_available_power(for_discharge=True)
        
        # Split capacity between Reg Up and RRS
        reg_up_capacity = available * 0.5
        rrs_capacity = available * 0.5
        
        return BidDecision(
            energy_bid_mw=0.0,
            energy_price_offer=0.0,
            reg_up_bid_mw=reg_up_capacity,
            rrs_bid_mw=rrs_capacity,
            reg_up_price=reg_up_price * 0.9,  # Competitive offer
            rrs_price=rrs_price * 0.9,
        )
    
    def _bid_ancillary_services_rtcb(
        self,
        battery: BatterySimulator,
        market_data: pd.Series,
        energy_bid: BidDecision,
    ) -> BidDecision:
        """Generate AS bid for RTC+B (stackable with energy)."""
        reg_up_price = float(market_data.get("reg_up_price", 0))
        reg_down_price = float(market_data.get("reg_down_price", 0))
        rrs_price = float(market_data.get("rrs_price", 0))
        ecrs_price = float(market_data.get("ecrs_price", 0))
        
        # Calculate remaining capacity after energy position
        available_discharge = battery.get_available_power(for_discharge=True)
        available_charge = battery.get_available_power(for_discharge=False)
        
        energy_position = abs(energy_bid.energy_bid_mw)
        remaining_discharge = max(0, available_discharge - energy_position)
        remaining_charge = max(0, available_charge - energy_position)
        
        # Allocate remaining capacity to AS
        # Prioritize higher-value services
        reg_up_capacity = min(remaining_discharge * 0.3, remaining_discharge)
        rrs_capacity = min(remaining_discharge * 0.3, remaining_discharge - reg_up_capacity)
        ecrs_capacity = remaining_discharge - reg_up_capacity - rrs_capacity
        
        reg_down_capacity = remaining_charge * 0.3
        
        return BidDecision(
            energy_bid_mw=0.0,  # Already in energy_bid
            energy_price_offer=0.0,
            reg_up_bid_mw=reg_up_capacity,
            reg_down_bid_mw=reg_down_capacity,
            rrs_bid_mw=rrs_capacity,
            ecrs_bid_mw=ecrs_capacity,
            reg_up_price=reg_up_price * 0.9,
            reg_down_price=reg_down_price * 0.9,
            rrs_price=rrs_price * 0.9,
            ecrs_price=ecrs_price * 0.9,
        )


class PredictiveStrategy(BiddingStrategy):
    """
    Predictive strategy using price forecasts.
    
    Uses HayekNet's existing components:
    - EnKF for state estimation
    - Bayesian reasoning for uncertainty quantification
    - RL for learning optimal policies
    """
    
    def __init__(self):
        """Initialize predictive strategy."""
        # Placeholder for ML models
        self.price_forecast_model = None
        self.rl_policy = None
    
    def generate_bid(
        self,
        battery: BatterySimulator,
        market_data: pd.Series,
        market_design: MarketDesign,
    ) -> BidDecision:
        """Generate ML-informed bid (placeholder for future enhancement)."""
        # For now, fall back to simple arbitrage
        simple_strategy = SimpleArbitrageStrategy()
        return simple_strategy.generate_bid(battery, market_data, market_design)
