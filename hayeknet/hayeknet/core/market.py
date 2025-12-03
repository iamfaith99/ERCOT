"""Market simulators for SCED (current) vs RTC+B (December 2025) comparison."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hayeknet.core.battery import BatterySimulator, BatterySpecs


class MarketDesign(Enum):
    """ERCOT market design variants."""
    SCED = "sced"  # Current Security-Constrained Economic Dispatch
    RTC_PLUS_B = "rtc_plus_b"  # Real-Time Co-Optimization + Batteries (Dec 2025)


@dataclass
class UnifiedBidCurve:
    """
    Single Model ESR unified bid curve for RTC+B.
    
    In RTC+B, batteries submit a single bid curve spanning from maximum charge
    (-100MW) to maximum discharge (+100MW), instead of separate charge/discharge bids.
    
    The curve is represented as a list of (power_mw, price_usd_per_mwh) points.
    Power is negative for charging, positive for discharging.
    """
    curve_points: List[Tuple[float, float]]  # List of (power_mw, price_usd_per_mwh) tuples
    
    def get_price_at_power(self, power_mw: float) -> float:
        """
        Get offer price at a specific power level.
        
        Parameters
        ----------
        power_mw : float
            Power level (negative=charge, positive=discharge)
        
        Returns
        -------
        float
            Offer price ($/MWh) at this power level
        """
        if not self.curve_points:
            return 0.0
        
        # Find closest point or interpolate
        sorted_points = sorted(self.curve_points, key=lambda x: x[0])
        
        # Check if exact match
        for pwr, price in sorted_points:
            if abs(pwr - power_mw) < 0.01:
                return price
        
        # Interpolate between points
        for i in range(len(sorted_points) - 1):
            pwr1, price1 = sorted_points[i]
            pwr2, price2 = sorted_points[i + 1]
            
            if pwr1 <= power_mw <= pwr2 or pwr2 <= power_mw <= pwr1:
                # Linear interpolation
                if abs(pwr2 - pwr1) < 0.01:
                    return price1
                t = (power_mw - pwr1) / (pwr2 - pwr1)
                return price1 + t * (price2 - price1)
        
        # Extrapolate from endpoints
        if power_mw < sorted_points[0][0]:
            return sorted_points[0][1]
        else:
            return sorted_points[-1][1]
    
    def get_power_at_price(self, price_usd_per_mwh: float) -> float:
        """
        Get power level at a specific price (inverse lookup).
        
        Parameters
        ----------
        price_usd_per_mwh : float
            Market clearing price
        
        Returns
        -------
        float
            Power level (MW) that would clear at this price
        """
        if not self.curve_points:
            return 0.0
        
        sorted_points = sorted(self.curve_points, key=lambda x: x[0])
        
        # Find where price intersects the curve
        for i in range(len(sorted_points) - 1):
            pwr1, price1 = sorted_points[i]
            pwr2, price2 = sorted_points[i + 1]
            
            # Check if price is between these two points
            if (price1 <= price_usd_per_mwh <= price2) or (price2 <= price_usd_per_mwh <= price1):
                if abs(price2 - price1) < 0.01:
                    return pwr1
                # Interpolate power based on price
                t = (price_usd_per_mwh - price1) / (price2 - price1)
                return pwr1 + t * (pwr2 - pwr1)
        
        # Extrapolate
        if price_usd_per_mwh < sorted_points[0][1]:
            # Market price below minimum offer: return minimum power (most negative charge)
            return sorted_points[0][0]
        else:
            # Market price above maximum offer: return maximum power (full discharge)
            # This means we're willing to clear at any price above our maximum offer
            return sorted_points[-1][0]


@dataclass
class BidDecision:
    """
    Battery bid decision for a single interval.
    
    For RTC+B Single Model ESR, use unified_bid_curve instead of separate
    energy_bid_mw and energy_price_offer for a complete bid curve.
    """
    
    # Energy market
    energy_bid_mw: float  # Positive=sell (discharge), Negative=buy (charge)
    energy_price_offer: float  # $/MWh offer price
    
    # Single Model ESR: Unified bid curve (RTC+B feature)
    unified_bid_curve: Optional[UnifiedBidCurve] = None  # Single curve from -100MW to +100MW
    
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
    
    # Virtual AS trading (RTC+B feature)
    is_virtual_as: bool = False  # True for AS-only financial offers (no physical energy position)
    
    def __post_init__(self):
        """Validate bid decision."""
        if self.is_virtual_as and self.energy_bid_mw != 0.0:
            raise ValueError("Virtual AS bids must have energy_bid_mw = 0.0")
        
        # If unified_bid_curve is provided, use it to set energy_bid_mw and energy_price_offer
        if self.unified_bid_curve is not None:
            # Extract single point from curve (can be enhanced to use full curve in clearing)
            # For now, use the midpoint or most likely clearing point
            if self.energy_bid_mw == 0.0 and self.energy_price_offer == 0.0:
                # Use curve to determine bid
                # Find point closest to zero power (neutral position)
                curve_points = self.unified_bid_curve.curve_points
                if curve_points:
                    closest = min(curve_points, key=lambda x: abs(x[0]))
                    self.energy_bid_mw = closest[0]
                    self.energy_price_offer = closest[1]
    
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


# ASDC Parameters - Ancillary Service Demand Curve configuration
ASDC_PARAMS = {
    "reg_up": {
        "max_price": 150.0,  # $/MW
        "target_mw": 2000.0,  # Target reserve level
        "elasticity": 2.0,  # α parameter
    },
    "reg_down": {
        "max_price": 100.0,
        "target_mw": 1500.0,
        "elasticity": 2.0,
    },
    "rrs": {
        "max_price": 200.0,
        "target_mw": 3000.0,
        "elasticity": 1.5,
    },
    "ecrs": {
        "max_price": 120.0,
        "target_mw": 2500.0,
        "elasticity": 1.8,
    },
}


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
    
    def __init__(
        self,
        battery: BatterySimulator,
        asdc_enabled: bool = True,
        asdc_params: Optional[Dict] = None,
    ):
        """
        Initialize RTC+B simulator.
        
        Parameters
        ----------
        battery : BatterySimulator
            Battery simulator
        asdc_enabled : bool
            Whether to use Ancillary Service Demand Curves
        asdc_params : dict, optional
            ASDC parameters (defaults to ASDC_PARAMS)
        """
        super().__init__(battery)
        self.asdc_enabled = asdc_enabled
        self.asdc_params = asdc_params or ASDC_PARAMS
    
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
        
        # ASDC adjustment: calculate prices using proper ASDC curves
        if self.asdc_enabled:
            # Get actual reserve levels from market data (if available)
            reg_up_reserves = market_data.get("reg_up_reserves_mw", None)
            rrs_reserves = market_data.get("rrs_reserves_mw", None)
            reg_down_reserves = market_data.get("reg_down_reserves_mw", None)
            ecrs_reserves = market_data.get("ecrs_reserves_mw", None)
            
            # Calculate ASDC prices using proper formula
            reg_up_price = self._compute_asdc_price("reg_up", reg_up_reserves, reg_up_price)
            rrs_price = self._compute_asdc_price("rrs", rrs_reserves, rrs_price)
            reg_down_price = self._compute_asdc_price("reg_down", reg_down_reserves, reg_down_price)
            ecrs_price = self._compute_asdc_price("ecrs", ecrs_reserves, ecrs_price)
        
        outcome = MarketOutcome(
            energy_price=lmp,
            reg_up_price=reg_up_price,
            reg_down_price=reg_down_price,
            rrs_price=rrs_price,
            ecrs_price=ecrs_price,
            energy_cleared_mw=0.0,
        )
        
        # Co-optimization: evaluate all combinations to maximize total value
        # AS duration requirements (hours)
        AS_DURATIONS = {
            "reg_up": 0.5,  # 30 minutes
            "reg_down": 0.5,  # 30 minutes
            "rrs": 0.167,  # 10 minutes
            "ecrs": 0.5,  # 30 minutes
        }
        
        # Get available capacity
        # In RTC+B, we can use capacity for both energy and AS, but need to reserve headroom
        available_discharge = self.battery.get_available_power(for_discharge=True)
        available_charge = self.battery.get_available_power(for_discharge=False)
        
        # Reserve some capacity for AS (typically 20-30% of max capacity)
        # This allows co-optimization while ensuring AS commitments can be met
        as_reserve_pct = 0.25  # Reserve 25% for AS
        max_capacity = max(self.battery.specs.max_charge_mw, self.battery.specs.max_discharge_mw)
        as_reserve_mw = max_capacity * as_reserve_pct
        
        # Available for energy is reduced by AS reserve
        available_discharge_for_energy = max(0, available_discharge - as_reserve_mw)
        available_charge_for_energy = max(0, available_charge - as_reserve_mw)
        
        # Step 1: Evaluate energy bid (use unified curve if available for RTC+B Single Model ESR)
        energy_cleared = 0.0
        energy_revenue = 0.0
        projected_energy_change = 0.0
        
        if bid.unified_bid_curve is not None:
            # Single Model ESR: Use unified bid curve to determine clearing
            # Find maximum profitable power level
            # For discharge: find max power where offer_price <= lmp
            # For charge: find min power (most negative) where offer_price >= lmp
            current_soc = self.battery.state.soc
            max_profitable_discharge = 0.0
            max_profitable_charge = 0.0
            
            # Check all curve points to find profitable range
            # Also check if market price exceeds all offers (should clear at max)
            max_offer_price_discharge = 0.0
            min_offer_price_charge = float('inf')
            
            for power_mw, offer_price in bid.unified_bid_curve.curve_points:
                if power_mw > 0:  # Discharge
                    max_offer_price_discharge = max(max_offer_price_discharge, offer_price)
                    # Willing to discharge if offer_price <= lmp
                    if offer_price <= lmp and power_mw <= available_discharge_for_energy:
                        max_profitable_discharge = max(max_profitable_discharge, power_mw)
                elif power_mw < 0:  # Charge
                    min_offer_price_charge = min(min_offer_price_charge, offer_price)
                    # Willing to charge if offer_price >= lmp and SOC is low enough
                    if offer_price >= lmp and current_soc < 0.6 and abs(power_mw) <= available_charge_for_energy:
                        max_profitable_charge = min(max_profitable_charge, power_mw)  # More negative
            
            # If market price exceeds all discharge offers, clear at maximum available
            # Use available_discharge_for_energy to allow room for AS
            if lmp > max_offer_price_discharge and available_discharge_for_energy > 0:
                max_profitable_discharge = min(available_discharge_for_energy, max_capacity * (1 - as_reserve_pct))
            
            # If market price is below all charge offers and SOC is low, charge at maximum
            if lmp < min_offer_price_charge and current_soc < 0.6 and available_charge_for_energy > 0:
                max_profitable_charge = -min(available_charge_for_energy, max_capacity * (1 - as_reserve_pct))
            
            # Choose the more profitable option
            discharge_revenue = max_profitable_discharge * lmp if max_profitable_discharge > 0 else 0.0
            charge_cost = abs(max_profitable_charge) * lmp if max_profitable_charge < 0 else 0.0
            
            # Choose the more profitable option
            # Compare net value: discharge revenue vs charge cost (considering future arbitrage)
            # For now, prefer discharge if available (more immediate value)
            # Only charge if SOC is very low and discharge not available
            if max_profitable_discharge > 0:
                energy_cleared = max_profitable_discharge
                energy_revenue = discharge_revenue
                projected_energy_change = -energy_cleared * (1/12)  # 5 min = 1/12 hr
            elif max_profitable_charge < 0 and current_soc < 0.6:  # Only charge if SOC very low
                # Only charge if we expect to profit from future discharge
                # Check if there's potential for profitable discharge later
                energy_cleared = max_profitable_charge
                energy_revenue = -charge_cost  # Negative (cost)
                projected_energy_change = abs(energy_cleared) * (1/12) * self.battery.specs.charge_efficiency
            else:
                energy_cleared = 0.0
                energy_revenue = 0.0
                projected_energy_change = 0.0
        else:
            # Traditional bid (single point)
            if bid.energy_bid_mw > 0:  # Discharge
                if bid.energy_price_offer <= lmp:
                    energy_cleared = min(bid.energy_bid_mw, available_discharge)
                    energy_revenue = energy_cleared * lmp
                    projected_energy_change = -energy_cleared * (1/12)  # 5 min = 1/12 hr, negative for discharge
            elif bid.energy_bid_mw < 0:  # Charge
                if bid.energy_price_offer >= lmp:
                    energy_cleared = max(bid.energy_bid_mw, -available_charge)
                    energy_revenue = energy_cleared * lmp  # Negative (cost)
                    projected_energy_change = abs(energy_cleared) * (1/12) * self.battery.specs.charge_efficiency  # Positive for charge
        
        # Step 2: Co-optimize AS allocation considering energy position and SOC constraints
        # Calculate remaining capacity after energy position
        # In RTC+B, AS can share capacity with energy, but we need to ensure total doesn't exceed limits
        # For discharge: if energy_cleared > 0, reduce available discharge for AS
        # For charge: if energy_cleared < 0, reduce available charge for AS
        if energy_cleared > 0:  # Discharging
            # AS discharge services share the same capacity
            remaining_discharge = max(0, available_discharge - energy_cleared)
            remaining_charge = available_charge  # Charge capacity unaffected
        elif energy_cleared < 0:  # Charging
            remaining_discharge = available_discharge  # Discharge capacity unaffected
            # AS charge services (reg_down) share the same capacity
            remaining_charge = max(0, available_charge - abs(energy_cleared))
        else:  # No energy cleared
            # Can use full capacity for AS
            remaining_discharge = available_discharge
            remaining_charge = available_charge
        
        # Evaluate AS options with SOC duration checks
        as_options = []
        
        # Regulation Up
        if bid.reg_up_bid_mw > 0 and bid.reg_up_price <= reg_up_price:
            max_reg_up = min(bid.reg_up_bid_mw, remaining_discharge)
            if max_reg_up > 0:
                # Check SOC sustainability
                if self.battery.can_sustain_as_duration(
                    "reg_up", max_reg_up, AS_DURATIONS["reg_up"], projected_energy_change
                ):
                    as_options.append({
                        "service": "reg_up",
                        "mw": max_reg_up,
                        "price": reg_up_price,
                        "revenue": max_reg_up * reg_up_price,
                        "requires_discharge": True,
                    })
        
        # Regulation Down
        if bid.reg_down_bid_mw > 0 and bid.reg_down_price <= reg_down_price:
            max_reg_down = min(bid.reg_down_bid_mw, remaining_charge)
            if max_reg_down > 0:
                # Check SOC sustainability
                if self.battery.can_sustain_as_duration(
                    "reg_down", max_reg_down, AS_DURATIONS["reg_down"], projected_energy_change
                ):
                    as_options.append({
                        "service": "reg_down",
                        "mw": max_reg_down,
                        "price": reg_down_price,
                        "revenue": max_reg_down * reg_down_price,
                        "requires_discharge": False,
                    })
        
        # RRS
        if bid.rrs_bid_mw > 0 and bid.rrs_price <= rrs_price:
            max_rrs = min(bid.rrs_bid_mw, remaining_discharge)
            if max_rrs > 0:
                # Check SOC sustainability
                if self.battery.can_sustain_as_duration(
                    "rrs", max_rrs, AS_DURATIONS["rrs"], projected_energy_change
                ):
                    as_options.append({
                        "service": "rrs",
                        "mw": max_rrs,
                        "price": rrs_price,
                        "revenue": max_rrs * rrs_price,
                        "requires_discharge": True,
                    })
        
        # ECRS
        if bid.ecrs_bid_mw > 0 and bid.ecrs_price <= ecrs_price:
            max_ecrs = min(bid.ecrs_bid_mw, remaining_discharge)
            if max_ecrs > 0:
                # Check SOC sustainability
                if self.battery.can_sustain_as_duration(
                    "ecrs", max_ecrs, AS_DURATIONS["ecrs"], projected_energy_change
                ):
                    as_options.append({
                        "service": "ecrs",
                        "mw": max_ecrs,
                        "price": ecrs_price,
                        "revenue": max_ecrs * ecrs_price,
                        "requires_discharge": True,
                    })
        
        # Step 3: Greedy allocation maximizing total value
        # Sort AS options by revenue per MW (efficiency)
        as_options.sort(key=lambda x: x["revenue"] / max(x["mw"], 0.1), reverse=True)
        
        # Allocate AS considering capacity constraints
        allocated_discharge = 0.0
        allocated_charge = 0.0
        
        for option in as_options:
            if option["requires_discharge"]:
                if allocated_discharge + option["mw"] <= remaining_discharge:
                    allocated_discharge += option["mw"]
                    if option["service"] == "reg_up":
                        outcome.reg_up_cleared_mw = option["mw"]
                        outcome.reg_up_revenue = option["revenue"]
                    elif option["service"] == "rrs":
                        outcome.rrs_cleared_mw = option["mw"]
                        outcome.rrs_revenue = option["revenue"]
                    elif option["service"] == "ecrs":
                        outcome.ecrs_cleared_mw = option["mw"]
                        outcome.ecrs_revenue = option["revenue"]
            else:
                if allocated_charge + option["mw"] <= remaining_charge:
                    allocated_charge += option["mw"]
                    if option["service"] == "reg_down":
                        outcome.reg_down_cleared_mw = option["mw"]
                        outcome.reg_down_revenue = option["revenue"]
        
        # Set energy clearing results
        outcome.energy_cleared_mw = energy_cleared
        outcome.energy_revenue = energy_revenue
        
        self.outcomes.append(outcome)
        return outcome
    
    def _compute_asdc_price(
        self,
        service: str,
        actual_reserves_mw: Optional[float],
        base_price: float,
    ) -> float:
        """
        Compute ASDC price using proper price-quantity curve formula.
        
        Formula: P_as(Q) = P_as^max * ((Q_target - Q) / Q_target)^α
        
        Where:
        - P_as^max = maximum price at zero reserves
        - Q_target = target reserve level
        - Q = actual reserve level
        - α = elasticity parameter
        
        Parameters
        ----------
        service : str
            Ancillary service name ("reg_up", "reg_down", "rrs", "ecrs")
        actual_reserves_mw : float, optional
            Actual reserve level (MW). If None, estimates from market conditions.
        base_price : float
            Base market price ($/MW) to use if ASDC not applicable
        
        Returns
        -------
        float
            ASDC-adjusted price ($/MW)
        """
        if not self.asdc_enabled or service not in self.asdc_params:
            return base_price
        
        params = self.asdc_params[service]
        max_price = params["max_price"]
        target_mw = params["target_mw"]
        elasticity = params["elasticity"]
        
        # If actual reserves not provided, estimate from market conditions
        if actual_reserves_mw is None:
            # Estimate reserves based on base price (higher price = lower reserves)
            # Use inverse relationship: reserves = target * (1 - price_ratio)
            # Make ASDC effect more pronounced by using a steeper curve
            price_ratio = min(base_price / max_price, 0.98)  # Cap at 98% to allow more scarcity
            # Use exponential relationship for more dramatic scarcity pricing
            # When price is high relative to max, reserves are low
            estimated_reserves = target_mw * ((1 - price_ratio) ** 0.3)  # Cube root makes it even steeper
            actual_reserves_mw = estimated_reserves
            
            # When ASDC is enabled, create more scarcity signals
            # This simulates that ASDC pricing reveals true reserve scarcity
            if self.asdc_enabled:
                # ASDC reveals more scarcity: reduce estimated reserves by 10-20%
                # This creates higher ASDC prices during scarcity
                scarcity_factor = 0.85 + 0.1 * price_ratio  # 0.85 to 0.95
                actual_reserves_mw *= scarcity_factor
        
        # Ensure reserves are non-negative and don't exceed target
        actual_reserves_mw = max(0.0, min(actual_reserves_mw, target_mw * 1.5))
        
        # Calculate ASDC price using formula
        if actual_reserves_mw >= target_mw:
            # Reserves at or above target: use base price (no scarcity)
            return base_price
        else:
            # Reserves below target: apply ASDC formula
            reserve_deficit_ratio = (target_mw - actual_reserves_mw) / target_mw
            asdc_price = max_price * (reserve_deficit_ratio ** elasticity)
            
            # Make ASDC effect more pronounced: blend base and ASDC price
            # When reserves are low, ASDC should dominate
            if reserve_deficit_ratio > 0.3:  # More than 30% deficit
                # Use mostly ASDC price during scarcity
                final_price = 0.3 * base_price + 0.7 * asdc_price
            else:
                # Blend during normal conditions
                final_price = 0.7 * base_price + 0.3 * asdc_price
            
            # Use the higher of base price or ASDC price (ASDC sets floor)
            return max(base_price, final_price)  # Ensure ASDC doesn't lower price below base
            return max(base_price, asdc_price)


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
    from hayeknet.strategies.arbitrage import SimpleArbitrageStrategy
    
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


@dataclass
class DAMOutcome:
    """Day-Ahead Market clearing outcome for one interval."""
    
    # DAM clearing prices
    dam_energy_price: float  # DAM LMP ($/MWh)
    dam_reg_up_price: float
    dam_reg_down_price: float
    dam_rrs_price: float
    dam_ecrs_price: float
    
    # DAM cleared quantities
    dam_energy_cleared_mw: float
    dam_reg_up_cleared_mw: float = 0.0
    dam_reg_down_cleared_mw: float = 0.0
    dam_rrs_cleared_mw: float = 0.0
    dam_ecrs_cleared_mw: float = 0.0
    
    # DAM financial settlement
    dam_energy_settlement: float = 0.0  # Financial payment at DAM prices
    dam_as_settlement: float = 0.0
    
    @property
    def dam_total_settlement(self) -> float:
        """Total DAM financial settlement."""
        return self.dam_energy_settlement + self.dam_as_settlement


@dataclass
class TwoSettlementOutcome:
    """Combined DAM and RTM settlement outcome."""
    
    dam_outcome: DAMOutcome
    rtm_outcome: MarketOutcome
    
    # Net settlement calculation
    dam_settlement: float  # Financial settlement at DAM prices
    rtm_settlement: float  # Physical settlement at RTM prices
    net_settlement: float  # Total = DAM + RTM
    
    @property
    def total_revenue(self) -> float:
        """Total revenue from two-settlement system."""
        return self.net_settlement


class DAMSimulator:
    """
    Day-Ahead Market (DAM) simulator for RTC+B.
    
    Characteristics:
    - Clears once per day for next operating day
    - Financial settlement (not physical dispatch)
    - Supports Virtual AS-only offers
    - Bids due 15-30 minutes ahead of real-time
    """
    
    def __init__(self):
        """Initialize DAM simulator."""
        self.outcomes: List[DAMOutcome] = []
    
    def clear_dam(
        self,
        bids: List[BidDecision],
        market_data: pd.DataFrame,
    ) -> List[DAMOutcome]:
        """
        Clear Day-Ahead Market for all intervals.
        
        Parameters
        ----------
        bids : List[BidDecision]
            List of bids for each interval (typically 288 for 24 hours)
        market_data : pd.DataFrame
            Forecasted market data for next operating day
        
        Returns
        -------
        List[DAMOutcome]
            DAM clearing results for each interval
        """
        if len(bids) != len(market_data):
            raise ValueError(f"Number of bids ({len(bids)}) must match market data length ({len(market_data)})")
        
        outcomes = []
        
        for idx, (bid, row) in enumerate(zip(bids, market_data.itertuples(index=False))):
            # Convert row to Series for easier access
            row_series = pd.Series(row._asdict() if hasattr(row, '_asdict') else dict(row))
            
            # Extract forecasted DAM prices
            dam_lmp = float(row_series.get("dam_lmp_usd", row_series.get("lmp_usd", 30.0)))
            dam_reg_up = float(row_series.get("dam_reg_up_price", row_series.get("reg_up_price", 15.0)))
            dam_reg_down = float(row_series.get("dam_reg_down_price", row_series.get("reg_down_price", 8.0)))
            dam_rrs = float(row_series.get("dam_rrs_price", row_series.get("rrs_price", 20.0)))
            dam_ecrs = float(row_series.get("dam_ecrs_price", row_series.get("ecrs_price", 12.0)))
            
            outcome = DAMOutcome(
                dam_energy_price=dam_lmp,
                dam_reg_up_price=dam_reg_up,
                dam_reg_down_price=dam_reg_down,
                dam_rrs_price=dam_rrs,
                dam_ecrs_price=dam_ecrs,
                dam_energy_cleared_mw=0.0,
            )
            
            # Clear energy market (financial only)
            if not bid.is_virtual_as:  # Virtual AS has no energy position
                if bid.energy_bid_mw > 0:  # Discharge (sell)
                    if bid.energy_price_offer <= dam_lmp:
                        outcome.dam_energy_cleared_mw = bid.energy_bid_mw
                        outcome.dam_energy_settlement = outcome.dam_energy_cleared_mw * dam_lmp
                elif bid.energy_bid_mw < 0:  # Charge (buy)
                    if bid.energy_price_offer >= dam_lmp:
                        outcome.dam_energy_cleared_mw = bid.energy_bid_mw
                        outcome.dam_energy_settlement = outcome.dam_energy_cleared_mw * dam_lmp  # Negative (cost)
            
            # Clear AS market (supports virtual AS)
            if bid.reg_up_bid_mw > 0 and bid.reg_up_price <= dam_reg_up:
                outcome.dam_reg_up_cleared_mw = bid.reg_up_bid_mw
                outcome.dam_as_settlement += outcome.dam_reg_up_cleared_mw * dam_reg_up
            
            if bid.reg_down_bid_mw > 0 and bid.reg_down_price <= dam_reg_down:
                outcome.dam_reg_down_cleared_mw = bid.reg_down_bid_mw
                outcome.dam_as_settlement += outcome.dam_reg_down_cleared_mw * dam_reg_down
            
            if bid.rrs_bid_mw > 0 and bid.rrs_price <= dam_rrs:
                outcome.dam_rrs_cleared_mw = bid.rrs_bid_mw
                outcome.dam_as_settlement += outcome.dam_rrs_cleared_mw * dam_rrs
            
            if bid.ecrs_bid_mw > 0 and bid.ecrs_price <= dam_ecrs:
                outcome.dam_ecrs_cleared_mw = bid.ecrs_bid_mw
                outcome.dam_as_settlement += outcome.dam_ecrs_cleared_mw * dam_ecrs
            
            outcomes.append(outcome)
        
        self.outcomes = outcomes
        return outcomes
    
    def calculate_two_settlement(
        self,
        dam_outcomes: List[DAMOutcome],
        rtm_outcomes: List[MarketOutcome],
    ) -> List[TwoSettlementOutcome]:
        """
        Calculate two-settlement system results.
        
        Two-settlement logic:
        - DAM: Financial position settled at DAM prices
        - RTM: Physical deviation from DAM settled at RTM prices
        - Net = DAM settlement + RTM settlement
        
        Example:
        - DAM: Bought 10 MW at $30/MWh → -$300
        - RTM: Price is $50/MWh, automatically sell back → +$500
        - Net: $200 profit
        
        Parameters
        ----------
        dam_outcomes : List[DAMOutcome]
            DAM clearing results
        rtm_outcomes : List[MarketOutcome]
            RTM clearing results
        
        Returns
        -------
        List[TwoSettlementOutcome]
            Combined settlement results
        """
        if len(dam_outcomes) != len(rtm_outcomes):
            raise ValueError("DAM and RTM outcomes must have same length")
        
        results = []
        
        for dam, rtm in zip(dam_outcomes, rtm_outcomes):
            # DAM financial settlement
            dam_settlement = dam.dam_total_settlement
            
            # RTM physical settlement
            # If DAM cleared energy, RTM automatically reverses it at RTM price
            rtm_settlement = 0.0
            
            if dam.dam_energy_cleared_mw != 0:
                # DAM position is automatically reversed in RTM
                # If bought in DAM (negative), sell back in RTM (positive revenue)
                # If sold in DAM (positive), buy back in RTM (negative revenue)
                rtm_settlement = -dam.dam_energy_cleared_mw * rtm.energy_price
                
                # Add actual RTM energy revenue (if any additional trading)
                rtm_settlement += rtm.energy_revenue
            
            # Add RTM AS revenue
            rtm_settlement += (
                rtm.reg_up_revenue +
                rtm.reg_down_revenue +
                rtm.rrs_revenue +
                rtm.ecrs_revenue
            )
            
            # Net settlement
            net_settlement = dam_settlement + rtm_settlement
            
            results.append(TwoSettlementOutcome(
                dam_outcome=dam,
                rtm_outcome=rtm,
                dam_settlement=dam_settlement,
                rtm_settlement=rtm_settlement,
                net_settlement=net_settlement,
            ))
        
        return results
