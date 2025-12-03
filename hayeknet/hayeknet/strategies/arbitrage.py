"""Simple arbitrage bidding strategy for batteries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hayeknet.core.battery import BatterySimulator
from hayeknet.core.market import BidDecision, MarketDesign, UnifiedBidCurve
from hayeknet.strategies.base import BiddingStrategy


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
            # RTC+B: Can do both energy and AS simultaneously
            # Strategy: Prioritize energy arbitrage when profitable, then add AS
            
            # Generate unified bid curve for Single Model ESR
            unified_curve = self.generate_unified_bid_curve(battery, market_data)
            
            # Generate energy bid first (prioritize profitable arbitrage)
            energy_bid = self._bid_energy_arbitrage(battery, lmp, price_low, price_high)
            
            # Calculate expected energy revenue to decide AS allocation
            # Only bid AS if energy arbitrage isn't highly profitable
            energy_profitability = 0.0
            if energy_bid.energy_bid_mw > 0:  # Discharge
                # Estimate profit from discharge
                energy_profitability = (lmp - 25.0) * energy_bid.energy_bid_mw  # Rough estimate
            elif energy_bid.energy_bid_mw < 0:  # Charge
                # Estimate profit from charge (negative, but we'll compare)
                energy_profitability = (lmp - 25.0) * abs(energy_bid.energy_bid_mw)  # Rough estimate
            
            # Generate AS bid, but be more conservative if energy is very profitable
            as_bid = self._bid_ancillary_services_rtcb(battery, market_data, energy_bid)
            
            # If energy is highly profitable, reduce AS bids to avoid capacity conflicts
            if energy_profitability > 5000:  # High energy profitability threshold
                # Reduce AS bids to 50% to prioritize energy
                as_bid.reg_up_bid_mw *= 0.5
                as_bid.rrs_bid_mw *= 0.5
                as_bid.ecrs_bid_mw *= 0.5
                as_bid.reg_down_bid_mw *= 0.5
            
            # Combine bids with unified curve
            bid = BidDecision(
                energy_bid_mw=energy_bid.energy_bid_mw,  # Fallback point
                energy_price_offer=energy_bid.energy_price_offer,
                unified_bid_curve=unified_curve,  # Single Model ESR curve
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
    
    def generate_unified_bid_curve(
        self,
        battery: BatterySimulator,
        market_data: pd.Series,
        num_points: int = 21,
    ) -> UnifiedBidCurve:
        """
        Generate Single Model ESR unified bid curve for RTC+B.
        
        Creates a single bid curve spanning from -100MW (max charge) to +100MW (max discharge).
        This is the key feature of RTC+B Single Model ESR, replacing separate charge/discharge bids.
        
        Parameters
        ----------
        battery : BatterySimulator
            Battery simulator
        market_data : pd.Series
            Market data for this interval
        num_points : int
            Number of points in the bid curve (default: 21, covering -100 to +100 MW)
        
        Returns
        -------
        UnifiedBidCurve
            Unified bid curve from -max_charge to +max_discharge
        """
        lmp = float(market_data.get("lmp_usd", 0))
        soc = battery.state.soc
        
        # Get battery capacity limits
        max_charge = battery.specs.max_charge_mw
        max_discharge = battery.specs.max_discharge_mw
        
        # Generate curve points from -max_charge to +max_discharge
        power_range = np.linspace(-max_charge, max_discharge, num_points)
        curve_points = []
        
        for power_mw in power_range:
            if power_mw < 0:  # Charging
                # Willingness to pay depends on:
                # 1. Current SOC (lower SOC = more willing to pay)
                # 2. Expected future prices
                # 3. Opportunity cost
                
                # More aggressive charging to enable more arbitrage
                if soc > 0.8:
                    # Very high SOC: conservative
                    offer_price = lmp * 0.6  # Willing to pay 60% of LMP
                elif soc > 0.6:
                    # Medium-high SOC: moderate
                    offer_price = lmp * (0.7 + 0.1 * (1.0 - soc))  # 70-80% of LMP
                else:
                    # Low SOC: aggressive charging
                    soc_factor = 1.0 - soc  # Higher when SOC is low
                    base_price = lmp * (0.75 + 0.15 * soc_factor)  # Willing to pay 75-90% of LMP
                    
                    # Adjust for charge capacity
                    charge_factor = abs(power_mw) / max_charge
                    offer_price = base_price * (1.0 + 0.03 * charge_factor)  # Up to 3% premium
                
            elif power_mw > 0:  # Discharging
                # Willingness to accept depends on:
                # 1. Current SOC (higher SOC = lower minimum price)
                # 2. Expected future prices
                # 3. Opportunity cost
                
                # Much more aggressive discharge to capture arbitrage value
                # Willing to accept lower prices to ensure we clear and make profit
                soc_factor = soc  # Higher when SOC is high
                # Willing to accept 70-90% of LMP (very aggressive to ensure clearing)
                base_price = lmp * (0.70 + 0.20 * soc_factor)
                
                # Adjust for discharge capacity - more aggressive at lower power levels
                discharge_factor = power_mw / max_discharge
                # More discount at lower power (more willing to clear)
                offer_price = base_price * (1.0 - 0.1 * (1 - discharge_factor))  # Up to 10% discount at low power
                
            else:  # Zero power
                offer_price = lmp
            
            curve_points.append((power_mw, offer_price))
        
        return UnifiedBidCurve(curve_points=curve_points)
    
    def generate_dam_bid(
        self,
        battery: BatterySimulator,
        market_data: pd.Series,
        forecast_uncertainty: float = 0.1,
    ) -> BidDecision:
        """
        Generate Day-Ahead Market bid.
        
        DAM bids are submitted 15-30 minutes ahead and must account for:
        - Forecast uncertainty (prices may differ from forecast)
        - Binding commitment (must be physically feasible)
        - Two-settlement risk (DAM vs RTM price differences)
        
        Parameters
        ----------
        battery : BatterySimulator
            Battery simulator
        market_data : pd.Series
            Forecasted market data for next interval
        forecast_uncertainty : float
            Expected forecast error (default: 10%)
        
        Returns
        -------
        BidDecision
            DAM bid decision
        """
        # Get forecasted DAM prices
        dam_lmp = float(market_data.get("dam_lmp_usd", market_data.get("lmp_usd", 30.0)))
        dam_reg_up = float(market_data.get("dam_reg_up_price", market_data.get("reg_up_price", 15.0)))
        dam_reg_down = float(market_data.get("dam_reg_down_price", market_data.get("reg_down_price", 8.0)))
        dam_rrs = float(market_data.get("dam_rrs_price", market_data.get("rrs_price", 20.0)))
        dam_ecrs = float(market_data.get("dam_ecrs_price", market_data.get("ecrs_price", 12.0)))
        
        # Adjust thresholds for forecast uncertainty (be more conservative)
        # Use wider thresholds to account for forecast error
        price_low = dam_lmp * (1 - forecast_uncertainty)
        price_high = dam_lmp * (1 + forecast_uncertainty)
        
        # Generate energy bid (more conservative due to forecast uncertainty)
        soc = battery.state.soc
        
        energy_bid_mw = 0.0
        energy_price_offer = dam_lmp
        
        if dam_lmp < price_low and soc < 0.75:  # More conservative SOC threshold
            charge_capacity = battery.get_available_power(for_discharge=False)
            energy_bid_mw = -charge_capacity * 0.8  # Reduce by 20% for uncertainty
            energy_price_offer = price_low * 1.05  # Slightly more conservative
        elif dam_lmp > price_high and soc > 0.25:  # More conservative SOC threshold
            discharge_capacity = battery.get_available_power(for_discharge=True)
            energy_bid_mw = discharge_capacity * 0.8  # Reduce by 20% for uncertainty
            energy_price_offer = price_high * 0.95  # Slightly more conservative
        
        # Generate AS bids (can be more aggressive as AS prices are more stable)
        reg_up_bid = 0.0
        reg_down_bid = 0.0
        rrs_bid = 0.0
        ecrs_bid = 0.0
        
        available_discharge = battery.get_available_power(for_discharge=True)
        available_charge = battery.get_available_power(for_discharge=False)
        
        # Allocate AS capacity (conservative allocation for DAM)
        if dam_reg_up > dam_lmp * 0.5:  # AS attractive if > 50% of energy price
            reg_up_bid = min(available_discharge * 0.3, available_discharge - abs(energy_bid_mw))
        
        if dam_rrs > dam_lmp * 0.6:
            remaining = available_discharge - abs(energy_bid_mw) - reg_up_bid
            rrs_bid = min(remaining * 0.3, remaining)
        
        if dam_reg_down > dam_lmp * 0.4:
            reg_down_bid = min(available_charge * 0.2, available_charge - abs(energy_bid_mw))
        
        return BidDecision(
            energy_bid_mw=energy_bid_mw,
            energy_price_offer=energy_price_offer,
            reg_up_bid_mw=reg_up_bid,
            reg_down_bid_mw=reg_down_bid,
            rrs_bid_mw=rrs_bid,
            ecrs_bid_mw=ecrs_bid,
            reg_up_price=dam_reg_up * 0.95,
            reg_down_price=dam_reg_down * 0.95,
            rrs_price=dam_rrs * 0.95,
            ecrs_price=dam_ecrs * 0.95,
            is_virtual_as=False,
        )
    
    def generate_virtual_as_bid(
        self,
        market_data: pd.Series,
        expected_rtm_as_spread: float = 1.2,
    ) -> BidDecision:
        """
        Generate Virtual AS-only offer for Day-Ahead Market.
        
        Virtual AS offers are financial positions only (no physical energy).
        Profitable when:
        - DAM AS prices < expected RTM AS prices
        - AS spread is attractive (expected_rtm_as_spread > 1.0)
        
        This is a new trading avenue in RTC+B that allows financial traders
        to speculate on AS prices without physical assets.
        
        Parameters
        ----------
        market_data : pd.Series
            Forecasted market data
        expected_rtm_as_spread : float
            Expected RTM AS price / DAM AS price ratio
            If > 1.0, RTM AS prices expected to be higher (profitable to buy in DAM)
        
        Returns
        -------
        BidDecision
            Virtual AS bid (energy_bid_mw = 0, is_virtual_as = True)
        """
        # Get DAM AS prices
        dam_reg_up = float(market_data.get("dam_reg_up_price", market_data.get("reg_up_price", 15.0)))
        dam_reg_down = float(market_data.get("dam_reg_down_price", market_data.get("reg_down_price", 8.0)))
        dam_rrs = float(market_data.get("dam_rrs_price", market_data.get("rrs_price", 20.0)))
        dam_ecrs = float(market_data.get("dam_ecrs_price", market_data.get("ecrs_price", 12.0)))
        
        # Expected RTM AS prices
        expected_rtm_reg_up = dam_reg_up * expected_rtm_as_spread
        expected_rtm_rrs = dam_rrs * expected_rtm_as_spread
        
        # Virtual AS bid: Offer to buy AS in DAM if expected to profit in RTM
        # This is a financial position - no physical energy required
        reg_up_bid = 0.0
        rrs_bid = 0.0
        
        # Only bid if spread is attractive (expected RTM > DAM)
        if expected_rtm_as_spread > 1.1:  # At least 10% spread
            # Typical virtual AS bid size (can be scaled)
            # In practice, this would be based on risk limits and capital
            base_virtual_size = 50.0  # MW (example)
            
            if expected_rtm_reg_up > dam_reg_up * 1.15:  # 15%+ spread
                reg_up_bid = base_virtual_size
            
            if expected_rtm_rrs > dam_rrs * 1.15:
                rrs_bid = base_virtual_size
        
        return BidDecision(
            energy_bid_mw=0.0,  # No energy position for virtual AS
            energy_price_offer=0.0,
            reg_up_bid_mw=reg_up_bid,
            reg_down_bid_mw=0.0,
            rrs_bid_mw=rrs_bid,
            ecrs_bid_mw=0.0,
            reg_up_price=dam_reg_up * 1.02,  # Willing to pay slightly above DAM
            reg_down_price=0.0,
            rrs_price=dam_rrs * 1.02,
            ecrs_price=0.0,
            is_virtual_as=True,  # Mark as virtual AS offer
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
