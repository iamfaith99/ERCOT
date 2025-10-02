"""Market simulators for SCED (current) vs RTC+B (December 2025) comparison."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import pandas as pd

from python.battery_model import BatterySimulator, BatterySpecs


class MarketDesign(Enum):
    """ERCOT market design variants."""
    SCED = "sced"  # Current Security-Constrained Economic Dispatch
    RTC_PLUS_B = "rtc_plus_b"  # Real-Time Co-Optimization + Batteries (Dec 2025)


@dataclass
class BidDecision:
    """Battery bid decision for a single interval."""
    
    # Energy market
    energy_bid_mw: float  # Positive=sell (discharge), Negative=buy (charge)
    energy_price_offer: float  # $/MWh offer price
    
    # Ancillary services (AS)
    reg_up_bid_mw: float = 0.0
    reg_down_bid_mw: float = 0.0
    rrs_bid_mw: float = 0.0  # Responsive reserve
    ecrs_bid_mw: float = 0.0  # ERCOT contingency reserve
    
    # AS price offers
    reg_up_price: float = 0.0
    reg_down_price: float = 0.0
    rrs_price: float = 0.0
    ecrs_price: float = 0.0
    
    @property
    def total_energy_position_mw(self) -> float:
        """Total energy position including AS reserves."""
        return self.energy_bid_mw + self.reg_up_bid_mw + self.rrs_bid_mw + self.ecrs_bid_mw


@dataclass
class MarketOutcome:
    """Outcome of market clearing for one interval."""
    
    # Prices
    energy_price: float  # LMP ($/MWh)
    reg_up_price: float
    reg_down_price: float
    rrs_price: float
    ecrs_price: float
    
    # Cleared quantities
    energy_cleared_mw: float
    reg_up_cleared_mw: float = 0.0
    reg_down_cleared_mw: float = 0.0
    rrs_cleared_mw: float = 0.0
    ecrs_cleared_mw: float = 0.0
    
    # Revenue components
    energy_revenue: float = 0.0
    reg_up_revenue: float = 0.0
    reg_down_revenue: float = 0.0
    rrs_revenue: float = 0.0
    ecrs_revenue: float = 0.0
    
    @property
    def total_revenue(self) -> float:
        """Total revenue from all markets."""
        return (
            self.energy_revenue +
            self.reg_up_revenue +
            self.reg_down_revenue +
            self.rrs_revenue +
            self.ecrs_revenue
        )


class MarketSimulator(ABC):
    """Abstract base class for market simulators."""
    
    def __init__(self, battery: BatterySimulator):
        """
        Initialize market simulator.
        
        Parameters
        ----------
        battery : BatterySimulator
            Battery simulator instance
        """
        self.battery = battery
        self.outcomes: List[MarketOutcome] = []
    
    @abstractmethod
    def clear_market(
        self,
        bid: BidDecision,
        market_data: pd.Series,
    ) -> MarketOutcome:
        """
        Clear market and determine outcomes.
        
        Parameters
        ----------
        bid : BidDecision
            Battery bid decision
        market_data : pd.Series
            Market data for this interval (prices, load, etc.)
            
        Returns
        -------
        MarketOutcome
            Market clearing result
        """
        pass
    
    @abstractmethod
    def get_design_name(self) -> str:
        """Return market design name."""
        pass


class SCEDSimulator(MarketSimulator):
    """
    Simulate current ERCOT SCED market design.
    
    Characteristics:
    - Energy and AS markets cleared separately
    - AS awards based on merit order without energy co-optimization
    - Batteries bid as separate charge/discharge resources
    - No explicit battery model in dispatch
    """
    
    def get_design_name(self) -> str:
        return "SCED (Current)"
    
    def clear_market(
        self,
        bid: BidDecision,
        market_data: pd.Series,
    ) -> MarketOutcome:
        """
        Simulate SCED market clearing.
        
        In SCED:
        1. Energy market clears first
        2. AS markets clear independently
        3. No co-optimization across markets
        4. Battery must choose: energy arbitrage OR AS
        """
        # Extract market prices with fallbacks for historical data
        lmp = float(market_data.get("lmp_usd", market_data.get("LMP", 25.0)))
        reg_up_price = float(market_data.get("reg_up_price", 15.0))
        reg_down_price = float(market_data.get("reg_down_price", 8.0))
        rrs_price = float(market_data.get("rrs_price", 20.0))
        ecrs_price = float(market_data.get("ecrs_price", 12.0))
        
        # Log data sources for debugging
        data_source = market_data.get("data_source", "unknown")
        if hasattr(self, '_log_data_sources') and getattr(self, '_log_data_sources', False):
            print(f"📊 SCED using data from: {data_source} (LMP: ${lmp:.2f}/MWh, RegUp: ${reg_up_price:.2f}/MW)")
        
        outcome = MarketOutcome(
            energy_price=lmp,
            reg_up_price=reg_up_price,
            reg_down_price=reg_down_price,
            rrs_price=rrs_price,
            ecrs_price=ecrs_price,
            energy_cleared_mw=0.0,  # Will be set below
        )
        
        # Energy market clearing (simplified: accept all economic bids)
        if bid.energy_bid_mw > 0:  # Discharge (sell)
            if bid.energy_price_offer <= lmp:
                outcome.energy_cleared_mw = bid.energy_bid_mw
                outcome.energy_revenue = outcome.energy_cleared_mw * lmp
            else:
                outcome.energy_cleared_mw = 0.0
        elif bid.energy_bid_mw < 0:  # Charge (buy)
            if bid.energy_price_offer >= lmp:
                outcome.energy_cleared_mw = bid.energy_bid_mw
                outcome.energy_revenue = outcome.energy_cleared_mw * lmp  # Negative (cost)
            else:
                outcome.energy_cleared_mw = 0.0
        else:
            outcome.energy_cleared_mw = 0.0
        
        # AS market clearing (simplified: accept if offering below market price)
        # In SCED, battery can't provide both energy and AS simultaneously well
        if outcome.energy_cleared_mw == 0:  # Only clear AS if not in energy market
            if bid.reg_up_bid_mw > 0 and bid.reg_up_price <= reg_up_price:
                outcome.reg_up_cleared_mw = bid.reg_up_bid_mw
                outcome.reg_up_revenue = outcome.reg_up_cleared_mw * reg_up_price
            
            if bid.reg_down_bid_mw > 0 and bid.reg_down_price <= reg_down_price:
                outcome.reg_down_cleared_mw = bid.reg_down_bid_mw
                outcome.reg_down_revenue = outcome.reg_down_cleared_mw * reg_down_price
            
            if bid.rrs_bid_mw > 0 and bid.rrs_price <= rrs_price:
                outcome.rrs_cleared_mw = bid.rrs_bid_mw
                outcome.rrs_revenue = outcome.rrs_cleared_mw * rrs_price
            
            if bid.ecrs_bid_mw > 0 and bid.ecrs_price <= ecrs_price:
                outcome.ecrs_cleared_mw = bid.ecrs_bid_mw
                outcome.ecrs_revenue = outcome.ecrs_cleared_mw * ecrs_price
        
        self.outcomes.append(outcome)
        return outcome


class RTCPlusBSimulator(MarketSimulator):
    """
    Simulate RTC+B market design (post-December 2025).
    
    Characteristics:
    - Co-optimization of energy and AS every 5 minutes
    - Explicit battery model with SOC constraints
    - Ancillary Service Demand Curves (ASDCs) integrated
    - Can provide energy and AS simultaneously with proper headroom
    - More efficient price formation
    """
    
    def __init__(self, battery: BatterySimulator, asdc_enabled: bool = True):
        """
        Initialize RTC+B simulator.
        
        Parameters
        ----------
        battery : BatterySimulator
            Battery simulator
        asdc_enabled : bool
            Whether to use Ancillary Service Demand Curves
        """
        super().__init__(battery)
        self.asdc_enabled = asdc_enabled
    
    def get_design_name(self) -> str:
        return "RTC+B (Dec 2025)"
    
    def clear_market(
        self,
        bid: BidDecision,
        market_data: pd.Series,
    ) -> MarketOutcome:
        """
        Simulate RTC+B market clearing with co-optimization.
        
        In RTC+B:
        1. Energy and AS co-optimized
        2. Battery can provide both simultaneously
        3. SOC constraints respected in dispatch
        4. ASDCs allow more efficient AS procurement
        """
        # Extract market prices with fallbacks for historical data
        lmp = float(market_data.get("lmp_usd", market_data.get("LMP", 25.0)))
        reg_up_price = float(market_data.get("reg_up_price", 15.0))
        reg_down_price = float(market_data.get("reg_down_price", 8.0))
        rrs_price = float(market_data.get("rrs_price", 20.0))
        ecrs_price = float(market_data.get("ecrs_price", 12.0))
        
        # Log data sources for debugging
        data_source = market_data.get("data_source", "unknown")
        if hasattr(self, '_log_data_sources') and getattr(self, '_log_data_sources', False):
            print(f"🗺 RTC+B using data from: {data_source} (LMP: ${lmp:.2f}/MWh, RegUp: ${reg_up_price:.2f}/MW)")
        
        # ASDC adjustment: prices more responsive to scarcity
        if self.asdc_enabled:
            scarcity_multiplier = self._compute_scarcity_multiplier(market_data)
            reg_up_price *= scarcity_multiplier
            rrs_price *= scarcity_multiplier
        
        outcome = MarketOutcome(
            energy_price=lmp,
            reg_up_price=reg_up_price,
            reg_down_price=reg_down_price,
            rrs_price=rrs_price,
            ecrs_price=ecrs_price,
            energy_cleared_mw=0.0,
        )
        
        # Co-optimization: battery can participate in multiple markets
        # Check energy bid
        if bid.energy_bid_mw > 0:  # Discharge
            if bid.energy_price_offer <= lmp:
                outcome.energy_cleared_mw = bid.energy_bid_mw
                outcome.energy_revenue = outcome.energy_cleared_mw * lmp
        elif bid.energy_bid_mw < 0:  # Charge
            if bid.energy_price_offer >= lmp:
                outcome.energy_cleared_mw = bid.energy_bid_mw
                outcome.energy_revenue = outcome.energy_cleared_mw * lmp
        else:
            outcome.energy_cleared_mw = 0.0
        
        # AS clearing (can stack with energy in RTC+B)
        available_discharge = self.battery.get_available_power(for_discharge=True)
        available_charge = self.battery.get_available_power(for_discharge=False)
        
        # Regulation Up (requires discharge headroom)
        if bid.reg_up_bid_mw > 0 and bid.reg_up_price <= reg_up_price:
            max_reg_up = min(bid.reg_up_bid_mw, available_discharge - abs(outcome.energy_cleared_mw))
            if max_reg_up > 0:
                outcome.reg_up_cleared_mw = max_reg_up
                outcome.reg_up_revenue = outcome.reg_up_cleared_mw * reg_up_price
        
        # Regulation Down (requires charge headroom)
        if bid.reg_down_bid_mw > 0 and bid.reg_down_price <= reg_down_price:
            max_reg_down = min(bid.reg_down_bid_mw, available_charge - abs(outcome.energy_cleared_mw))
            if max_reg_down > 0:
                outcome.reg_down_cleared_mw = max_reg_down
                outcome.reg_down_revenue = outcome.reg_down_cleared_mw * reg_down_price
        
        # RRS (spinning reserve, requires discharge capability)
        if bid.rrs_bid_mw > 0 and bid.rrs_price <= rrs_price:
            remaining_discharge = available_discharge - abs(outcome.energy_cleared_mw) - outcome.reg_up_cleared_mw
            max_rrs = min(bid.rrs_bid_mw, remaining_discharge)
            if max_rrs > 0:
                outcome.rrs_cleared_mw = max_rrs
                outcome.rrs_revenue = outcome.rrs_cleared_mw * rrs_price
        
        # ECRS (non-spinning reserve)
        if bid.ecrs_bid_mw > 0 and bid.ecrs_price <= ecrs_price:
            remaining_discharge = (
                available_discharge - abs(outcome.energy_cleared_mw) - 
                outcome.reg_up_cleared_mw - outcome.rrs_cleared_mw
            )
            max_ecrs = min(bid.ecrs_bid_mw, remaining_discharge)
            if max_ecrs > 0:
                outcome.ecrs_cleared_mw = max_ecrs
                outcome.ecrs_revenue = outcome.ecrs_cleared_mw * ecrs_price
        
        self.outcomes.append(outcome)
        return outcome
    
    def _compute_scarcity_multiplier(self, market_data: pd.Series) -> float:
        """
        Compute ASDC scarcity multiplier based on system conditions.
        
        Higher multiplier during tight conditions (high load, low reserves).
        Uses multiple indicators from multi-source data when available.
        """
        multiplier = 1.0
        
        # Load-based scarcity
        load_mw = market_data.get("net_load_mw") or market_data.get("total_load_mw")
        if load_mw:
            load = float(load_mw)
            if load > 70000:  # Very high load
                multiplier = max(multiplier, 1.5)
            elif load > 65000:  # High load
                multiplier = max(multiplier, 1.2)
        
        # Price-based scarcity indicator
        lmp = market_data.get("lmp_usd") or market_data.get("LMP")
        if lmp:
            price = float(lmp)
            if price > 100:  # High price indicates scarcity
                multiplier = max(multiplier, 1.4)
            elif price > 50:
                multiplier = max(multiplier, 1.1)
        
        # Generation mix indicators (if available from multi-source)
        renewable_output = market_data.get("renewable_output_pct")
        if renewable_output:
            renewable_pct = float(renewable_output)
            if renewable_pct < 0.2:  # Low renewable output = tight conditions
                multiplier = max(multiplier, 1.3)
        
        # Forecast error indicator (from load volatility)
        load_volatility = market_data.get("load_volatility")
        if load_volatility:
            volatility = float(load_volatility)
            if volatility > 3000:  # High volatility = uncertainty
                multiplier = max(multiplier, 1.2)
        
        return min(multiplier, 2.0)  # Cap at 2.0x


def compare_market_designs(
    market_data: pd.DataFrame,
    battery_specs: BatterySpecs,
    bidding_strategy: str = "simple_arbitrage",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run parallel simulations under SCED and RTC+B and compare outcomes.
    
    Parameters
    ----------
    market_data : pd.DataFrame
        Historical market data
    battery_specs : BatterySpecs
        Battery specifications
    bidding_strategy : str
        Bidding strategy to use ("simple_arbitrage", "as_focused", "hybrid")
        
    Returns
    -------
    comparison_df : pd.DataFrame
        Detailed comparison of outcomes
    summary : dict
        Summary statistics
    """
    from python.battery_strategy import SimpleArbitrageStrategy
    
    # Initialize batteries for each market design
    battery_sced = BatterySimulator(battery_specs)
    battery_rtcb = BatterySimulator(battery_specs)
    
    # Initialize market simulators
    sced_sim = SCEDSimulator(battery_sced)
    rtcb_sim = RTCPlusBSimulator(battery_rtcb, asdc_enabled=True)
    
    # Initialize strategy
    strategy = SimpleArbitrageStrategy()
    
    # Run simulations
    results_sced = []
    results_rtcb = []
    
    for idx, row in market_data.iterrows():
        # Generate bids for each design
        bid_sced = strategy.generate_bid(battery_sced, row, MarketDesign.SCED)
        bid_rtcb = strategy.generate_bid(battery_rtcb, row, MarketDesign.RTC_PLUS_B)
        
        # Clear markets
        outcome_sced = sced_sim.clear_market(bid_sced, row)
        outcome_rtcb = rtcb_sim.clear_market(bid_rtcb, row)
        
        # Execute battery operations
        battery_sced.step(
            outcome_sced.energy_cleared_mw,
            reg_up_mw=outcome_sced.reg_up_cleared_mw,
            rrs_mw=outcome_sced.rrs_cleared_mw,
        )
        battery_rtcb.step(
            outcome_rtcb.energy_cleared_mw,
            reg_up_mw=outcome_rtcb.reg_up_cleared_mw,
            reg_down_mw=outcome_rtcb.reg_down_cleared_mw,
            rrs_mw=outcome_rtcb.rrs_cleared_mw,
            ecrs_mw=outcome_rtcb.ecrs_cleared_mw,
        )
        
        # Record results
        results_sced.append({
            "timestamp": row.get("timestamp"),
            "revenue": outcome_sced.total_revenue,
            "energy_revenue": outcome_sced.energy_revenue,
            "as_revenue": outcome_sced.total_revenue - outcome_sced.energy_revenue,
            "soc": battery_sced.state.soc,
            "power_mw": battery_sced.state.power_mw,
            "market_design": "SCED",
        })
        
        results_rtcb.append({
            "timestamp": row.get("timestamp"),
            "revenue": outcome_rtcb.total_revenue,
            "energy_revenue": outcome_rtcb.energy_revenue,
            "as_revenue": outcome_rtcb.total_revenue - outcome_rtcb.energy_revenue,
            "soc": battery_rtcb.state.soc,
            "power_mw": battery_rtcb.state.power_mw,
            "market_design": "RTC+B",
        })
    
    # Combine results
    comparison_df = pd.DataFrame(results_sced + results_rtcb)
    
    # Compute summary statistics
    sced_total = sum(r["revenue"] for r in results_sced)
    rtcb_total = sum(r["revenue"] for r in results_rtcb)
    
    summary = {
        "sced_total_revenue": sced_total,
        "rtcb_total_revenue": rtcb_total,
        "revenue_improvement": rtcb_total - sced_total,
        "revenue_improvement_pct": ((rtcb_total - sced_total) / sced_total * 100) if sced_total > 0 else 0,
        "sced_cycles": battery_sced.state.total_cycles,
        "rtcb_cycles": battery_rtcb.state.total_cycles,
        "sced_as_revenue_pct": (sum(r["as_revenue"] for r in results_sced) / sced_total * 100) if sced_total > 0 else 0,
        "rtcb_as_revenue_pct": (sum(r["as_revenue"] for r in results_rtcb) / rtcb_total * 100) if rtcb_total > 0 else 0,
    }
    
    return comparison_df, summary
