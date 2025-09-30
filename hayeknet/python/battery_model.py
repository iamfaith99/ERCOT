"""Battery energy storage system (BESS) model for ERCOT trading simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class BatterySpecs:
    """Technical specifications for a battery energy storage system."""
    
    # Power capacity
    max_charge_mw: float = 100.0  # Maximum charging power (MW)
    max_discharge_mw: float = 100.0  # Maximum discharging power (MW)
    
    # Energy capacity
    capacity_mwh: float = 200.0  # Total energy storage capacity (MWh)
    min_soc: float = 0.1  # Minimum state of charge (fraction, 0-1)
    max_soc: float = 0.9  # Maximum state of charge (fraction, 0-1)
    initial_soc: float = 0.5  # Initial state of charge (fraction, 0-1)
    
    # Efficiency
    charge_efficiency: float = 0.95  # Charging efficiency (0-1)
    discharge_efficiency: float = 0.95  # Discharging efficiency (0-1)
    round_trip_efficiency: float = 0.9  # Overall round-trip efficiency
    self_discharge_rate: float = 0.0001  # Self-discharge per hour (fraction)
    
    # Economic parameters
    degradation_cost_per_mwh: float = 5.0  # Battery degradation cost ($/MWh)
    om_cost_per_mw: float = 2.0  # Operations & maintenance ($/MW)
    
    # Operational constraints
    ramp_rate_mw_per_min: float = 50.0  # Maximum ramp rate
    min_runtime_intervals: int = 1  # Minimum continuous operation
    max_cycles_per_day: float = 2.0  # Degradation constraint
    
    # Ancillary service participation
    can_provide_reg_up: bool = True
    can_provide_reg_down: bool = True
    can_provide_rrs: bool = True  # Responsive reserve
    can_provide_ecrs: bool = True  # ERCOT contingency reserve
    
    # Ancillary service headroom requirements
    reg_headroom_fraction: float = 0.1  # Reserve 10% for regulation
    
    @property
    def usable_capacity_mwh(self) -> float:
        """Usable energy capacity between SOC limits."""
        return self.capacity_mwh * (self.max_soc - self.min_soc)
    
    @property
    def combined_efficiency(self) -> float:
        """Combined charge-discharge efficiency."""
        return self.charge_efficiency * self.discharge_efficiency


@dataclass
class BatteryState:
    """Current operational state of a battery."""
    
    soc: float  # State of charge (fraction, 0-1)
    power_mw: float  # Current power output (MW, positive=discharge, negative=charge)
    energy_mwh: float  # Absolute energy level (MWh)
    cycles_today: float  # Number of equivalent full cycles completed today
    
    # Ancillary service commitments
    reg_up_commitment_mw: float = 0.0
    reg_down_commitment_mw: float = 0.0
    rrs_commitment_mw: float = 0.0
    ecrs_commitment_mw: float = 0.0
    
    # Cumulative metrics
    total_cycles: float = 0.0
    total_energy_charged_mwh: float = 0.0
    total_energy_discharged_mwh: float = 0.0
    total_degradation_cost: float = 0.0
    
    def copy(self) -> BatteryState:
        """Create a copy of the current state."""
        return BatteryState(
            soc=self.soc,
            power_mw=self.power_mw,
            energy_mwh=self.energy_mwh,
            cycles_today=self.cycles_today,
            reg_up_commitment_mw=self.reg_up_commitment_mw,
            reg_down_commitment_mw=self.reg_down_commitment_mw,
            rrs_commitment_mw=self.rrs_commitment_mw,
            ecrs_commitment_mw=self.ecrs_commitment_mw,
            total_cycles=self.total_cycles,
            total_energy_charged_mwh=self.total_energy_charged_mwh,
            total_energy_discharged_mwh=self.total_energy_discharged_mwh,
            total_degradation_cost=self.total_degradation_cost,
        )


class BatterySimulator:
    """Simulator for battery energy storage system operations."""
    
    def __init__(self, specs: BatterySpecs):
        """
        Initialize battery simulator.
        
        Parameters
        ----------
        specs : BatterySpecs
            Battery specifications
        """
        self.specs = specs
        self.state = self._initialize_state()
        self.history: list[BatteryState] = []
    
    def _initialize_state(self) -> BatteryState:
        """Initialize battery to starting conditions."""
        initial_energy = self.specs.capacity_mwh * self.specs.initial_soc
        return BatteryState(
            soc=self.specs.initial_soc,
            power_mw=0.0,
            energy_mwh=initial_energy,
            cycles_today=0.0,
        )
    
    def reset(self, initial_soc: Optional[float] = None) -> BatteryState:
        """
        Reset battery to initial conditions.
        
        Parameters
        ----------
        initial_soc : float, optional
            Starting SOC (defaults to spec initial_soc)
            
        Returns
        -------
        BatteryState
            Initial state
        """
        if initial_soc is not None:
            self.specs.initial_soc = initial_soc
        self.state = self._initialize_state()
        self.history = [self.state.copy()]
        return self.state
    
    def get_available_power(self, for_discharge: bool = True) -> float:
        """
        Get currently available power considering SOC and AS commitments.
        
        Parameters
        ----------
        for_discharge : bool
            If True, return discharge capacity; if False, return charge capacity
            
        Returns
        -------
        float
            Available power (MW)
        """
        if for_discharge:
            # Discharge limited by: max discharge, SOC, and AS commitments
            max_by_spec = self.specs.max_discharge_mw
            max_by_soc = max(0, (self.state.soc - self.specs.min_soc) * self.specs.capacity_mwh / 0.0833)  # 5 min = 0.0833 hr
            as_reserve = (
                self.state.reg_up_commitment_mw +
                self.state.rrs_commitment_mw +
                self.state.ecrs_commitment_mw
            )
            return max(0, min(max_by_spec, max_by_soc) - as_reserve)
        else:
            # Charge limited by: max charge, SOC, and AS commitments
            max_by_spec = self.specs.max_charge_mw
            max_by_soc = max(0, (self.specs.max_soc - self.state.soc) * self.specs.capacity_mwh / 0.0833)
            as_reserve = self.state.reg_down_commitment_mw
            return max(0, min(max_by_spec, max_by_soc) - as_reserve)
    
    def step(
        self,
        power_command_mw: float,
        interval_hours: float = 1/12,  # 5 minutes
        reg_up_mw: float = 0.0,
        reg_down_mw: float = 0.0,
        rrs_mw: float = 0.0,
        ecrs_mw: float = 0.0,
    ) -> Tuple[BatteryState, float]:
        """
        Simulate one time step of battery operation.
        
        Parameters
        ----------
        power_command_mw : float
            Commanded power (MW, positive=discharge, negative=charge)
        interval_hours : float
            Time step duration (hours, default 5 min = 1/12 hr)
        reg_up_mw : float
            Regulation up commitment (MW)
        reg_down_mw : float
            Regulation down commitment (MW)
        rrs_mw : float
            Responsive reserve commitment (MW)
        ecrs_mw : float
            Contingency reserve commitment (MW)
            
        Returns
        -------
        state : BatteryState
            New state after time step
        degradation_cost : float
            Degradation cost incurred ($)
        """
        # Store AS commitments
        self.state.reg_up_commitment_mw = reg_up_mw
        self.state.reg_down_commitment_mw = reg_down_mw
        self.state.rrs_commitment_mw = rrs_mw
        self.state.ecrs_commitment_mw = ecrs_mw
        
        # Check feasibility
        if power_command_mw > 0:  # Discharge
            available = self.get_available_power(for_discharge=True)
            actual_power = min(power_command_mw, available)
        else:  # Charge
            available = self.get_available_power(for_discharge=False)
            actual_power = max(power_command_mw, -available)
        
        # Calculate energy change with efficiency
        if actual_power > 0:  # Discharging
            energy_delivered = actual_power * interval_hours
            energy_from_battery = energy_delivered / self.specs.discharge_efficiency
            self.state.total_energy_discharged_mwh += energy_delivered
        elif actual_power < 0:  # Charging
            energy_consumed = -actual_power * interval_hours
            energy_to_battery = energy_consumed * self.specs.charge_efficiency
            self.state.total_energy_charged_mwh += energy_consumed
            energy_from_battery = -energy_to_battery
        else:
            energy_from_battery = 0.0
        
        # Apply self-discharge
        self_discharge = self.state.energy_mwh * self.specs.self_discharge_rate * interval_hours
        
        # Update energy and SOC
        new_energy = self.state.energy_mwh - energy_from_battery - self_discharge
        new_energy = np.clip(
            new_energy,
            self.specs.min_soc * self.specs.capacity_mwh,
            self.specs.max_soc * self.specs.capacity_mwh,
        )
        new_soc = new_energy / self.specs.capacity_mwh
        
        # Track cycles (one cycle = charge from min to max)
        energy_throughput = abs(energy_from_battery)
        cycle_increment = energy_throughput / self.specs.usable_capacity_mwh
        self.state.cycles_today += cycle_increment
        self.state.total_cycles += cycle_increment
        
        # Calculate degradation cost
        degradation_cost = energy_throughput * self.specs.degradation_cost_per_mwh
        self.state.total_degradation_cost += degradation_cost
        
        # Update state
        self.state.soc = new_soc
        self.state.energy_mwh = new_energy
        self.state.power_mw = actual_power
        
        # Record history
        self.history.append(self.state.copy())
        
        return self.state, degradation_cost
    
    def compute_arbitrage_value(
        self,
        charge_price: float,
        discharge_price: float,
        charge_duration_hours: float = 1.0,
    ) -> float:
        """
        Compute expected arbitrage profit from charge/discharge cycle.
        
        Parameters
        ----------
        charge_price : float
            Price during charging ($/MWh)
        discharge_price : float
            Price during discharging ($/MWh)
        charge_duration_hours : float
            Duration to charge (hours)
            
        Returns
        -------
        float
            Expected profit ($)
        """
        # Energy that can be charged
        charge_energy = min(
            self.specs.max_charge_mw * charge_duration_hours,
            (self.specs.max_soc - self.state.soc) * self.specs.capacity_mwh,
        )
        
        # Energy that can be discharged (accounting for efficiency)
        discharge_energy = charge_energy * self.specs.round_trip_efficiency
        
        # Revenue from discharge
        revenue = discharge_energy * discharge_price
        
        # Cost of charging
        cost = charge_energy * charge_price
        
        # Degradation cost
        degradation = (charge_energy + discharge_energy) * self.specs.degradation_cost_per_mwh
        
        # Net profit
        profit = revenue - cost - degradation
        
        return profit
    
    def can_participate_in_ancillary(self, service: str) -> bool:
        """Check if battery can participate in a specific ancillary service."""
        service = service.lower()
        if service == "reg_up":
            return self.specs.can_provide_reg_up and self.state.soc > self.specs.min_soc + 0.1
        elif service == "reg_down":
            return self.specs.can_provide_reg_down and self.state.soc < self.specs.max_soc - 0.1
        elif service == "rrs":
            return self.specs.can_provide_rrs and self.state.soc > self.specs.min_soc + 0.1
        elif service == "ecrs":
            return self.specs.can_provide_ecrs and self.state.soc > self.specs.min_soc + 0.1
        return False
    
    def get_state_summary(self) -> dict:
        """Get current state summary as dictionary."""
        return {
            "soc": self.state.soc,
            "soc_pct": self.state.soc * 100,
            "energy_mwh": self.state.energy_mwh,
            "power_mw": self.state.power_mw,
            "cycles_today": self.state.cycles_today,
            "total_cycles": self.state.total_cycles,
            "total_charged_mwh": self.state.total_energy_charged_mwh,
            "total_discharged_mwh": self.state.total_energy_discharged_mwh,
            "total_degradation_cost": self.state.total_degradation_cost,
            "available_discharge_mw": self.get_available_power(for_discharge=True),
            "available_charge_mw": self.get_available_power(for_discharge=False),
        }
