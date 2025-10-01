"""Battery-specific daily analysis for graduate research.

This module provides battery trading simulation and analysis
for the daily research workflow.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from python.battery_model import BatterySimulator, BatterySpecs
from python.battery_strategy import SimpleArbitrageStrategy
from python.market_simulator import MarketDesign


class BatteryDailyAnalyzer:
    """Analyze battery performance for daily research journal."""
    
    def __init__(self, specs: BatterySpecs | None = None):
        """Initialize with battery specifications.
        
        Parameters
        ----------
        specs : BatterySpecs, optional
            Battery specifications. If None, uses 100MW/400MWh default.
        """
        if specs is None:
            # Research default: 100 MW / 400 MWh
            specs = BatterySpecs(
                max_charge_mw=100.0,
                max_discharge_mw=100.0,
                capacity_mwh=400.0,
                initial_soc=0.5
            )
        
        self.specs = specs
        self.simulator = BatterySimulator(specs)
        self.strategy = SimpleArbitrageStrategy()
    
    def run_arbitrage_simulation(
        self, 
        lmp_data: pd.DataFrame,
        market_design: MarketDesign = MarketDesign.SCED
    ) -> Dict[str, Any]:
        """
        Run battery arbitrage simulation on LMP data.
        
        Parameters
        ----------
        lmp_data : pd.DataFrame
            LMP data with columns: timestamp, lmp_usd, settlement_point
        market_design : MarketDesign
            Market design to simulate (SCED or RTC_B)
            
        Returns
        -------
        dict
            Simulation results with SOC, actions, PnL, etc.
        """
        # Reset battery to initial state
        self.simulator.reset(initial_soc=self.specs.initial_soc)
        
        # Aggregate to system-average LMP by timestamp
        if 'settlement_point' in lmp_data.columns:
            # Average across all hubs for system price
            system_lmp = lmp_data.groupby('timestamp')['lmp_usd'].mean().reset_index()
        else:
            system_lmp = lmp_data[['timestamp', 'lmp_usd']].copy()
        
        system_lmp = system_lmp.sort_values('timestamp').reset_index(drop=True)
        
        # Simulation tracking
        results = {
            'timestamps': [],
            'lmp': [],
            'soc': [],
            'power_mw': [],
            'action': [],  # 'charge', 'discharge', 'idle'
            'energy_mwh': [],
            'revenue': [],
            'cumulative_pnl': [],
        }
        
        cumulative_pnl = 0.0
        
        for idx, row in system_lmp.iterrows():
            timestamp = row['timestamp']
            lmp = row['lmp_usd']
            
            # Generate bid decision
            market_data = pd.Series({
                'timestamp': timestamp,
                'lmp': lmp,
                'reg_up_mcpc': lmp * 0.5,  # Simplified: assume AS prices
                'reg_down_mcpc': lmp * 0.3,
                'rrs_mcpc': lmp * 0.4,
            })
            
            bid = self.strategy.generate_bid(
                battery=self.simulator,
                market_data=market_data,
                market_design=market_design
            )
            
            # Execute bid (simplified - assume bid always clears)
            interval_hours = 5.0 / 60.0  # 5-minute intervals
            
            # Use step() method with power command
            # Positive = discharge, negative = charge
            new_state, degradation_cost = self.simulator.step(
                power_command_mw=bid.energy_bid_mw,
                interval_hours=interval_hours
            )
            
            if bid.energy_bid_mw < 0:  # Charging
                action = 'charge'
                energy_mwh = abs(bid.energy_bid_mw) * interval_hours
                cost = energy_mwh * lmp
                revenue_interval = -cost - degradation_cost  # Negative = cost
            elif bid.energy_bid_mw > 0:  # Discharging
                action = 'discharge'
                energy_mwh = bid.energy_bid_mw * interval_hours
                revenue_interval = energy_mwh * lmp - degradation_cost
            else:
                action = 'idle'
                revenue_interval = -degradation_cost
            
            cumulative_pnl += revenue_interval
            
            # Record results
            state = self.simulator.state
            results['timestamps'].append(timestamp)
            results['lmp'].append(lmp)
            results['soc'].append(state.soc)
            results['power_mw'].append(state.power_mw)
            results['action'].append(action)
            results['energy_mwh'].append(state.energy_mwh)
            results['revenue'].append(revenue_interval)
            results['cumulative_pnl'].append(cumulative_pnl)
        
        return results
    
    def compute_battery_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute battery performance metrics from simulation results.
        
        Parameters
        ----------
        results : dict
            Simulation results from run_arbitrage_simulation()
            
        Returns
        -------
        dict
            Battery performance metrics
        """
        soc_array = np.array(results['soc'])
        power_array = np.array(results['power_mw'])
        revenue_array = np.array(results['revenue'])
        
        # Count actions
        actions = results['action']
        n_charge = actions.count('charge')
        n_discharge = actions.count('discharge')
        n_idle = actions.count('idle')
        
        # Estimate cycles (simplified)
        # One full cycle = charge from min to max and discharge back
        soc_range = soc_array.max() - soc_array.min()
        estimated_cycles = soc_range / (self.specs.max_soc - self.specs.min_soc)
        
        metrics = {
            # SOC metrics
            'soc_mean': float(soc_array.mean()),
            'soc_std': float(soc_array.std()),
            'soc_min': float(soc_array.min()),
            'soc_max': float(soc_array.max()),
            'soc_range': float(soc_range),
            'soc_utilization_pct': float(soc_range / (self.specs.max_soc - self.specs.min_soc) * 100),
            
            # Action metrics
            'charge_intervals': n_charge,
            'discharge_intervals': n_discharge,
            'idle_intervals': n_idle,
            'active_pct': float((n_charge + n_discharge) / len(actions) * 100),
            
            # Power metrics
            'avg_charge_power_mw': float(power_array[power_array < 0].mean()) if n_charge > 0 else 0.0,
            'avg_discharge_power_mw': float(power_array[power_array > 0].mean()) if n_discharge > 0 else 0.0,
            'max_power_mw': float(abs(power_array).max()),
            
            # Cycle metrics
            'estimated_cycles': float(estimated_cycles),
            'cycles_per_day_rate': float(estimated_cycles / (len(actions) / 288)),  # 288 = intervals per day
            
            # Revenue metrics
            'total_revenue': float(revenue_array.sum()),
            'charge_cost': float(revenue_array[revenue_array < 0].sum()),
            'discharge_revenue': float(revenue_array[revenue_array > 0].sum()),
            'final_pnl': float(results['cumulative_pnl'][-1]),
            'avg_revenue_per_interval': float(revenue_array.mean()),
            
            # Efficiency
            'gross_profit_margin': 0.0,  # Will calculate below
        }
        
        # Calculate gross profit margin
        if metrics['charge_cost'] < 0:  # It's negative
            metrics['gross_profit_margin'] = float(
                (metrics['discharge_revenue'] + metrics['charge_cost']) / abs(metrics['charge_cost']) * 100
            )
        
        return metrics
    
    def generate_battery_journal_section(
        self,
        lmp_data: pd.DataFrame,
        results: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate battery-specific research journal section.
        
        Parameters
        ----------
        lmp_data : pd.DataFrame
            Original LMP data
        results : dict
            Simulation results
        metrics : dict
            Computed metrics
            
        Returns
        -------
        str
            Markdown-formatted journal section
        """
        today = datetime.now()
        
        # Market summary
        lmp_mean = lmp_data['lmp_usd'].mean()
        lmp_std = lmp_data['lmp_usd'].std()
        lmp_min = lmp_data['lmp_usd'].min()
        lmp_max = lmp_data['lmp_usd'].max()
        
        journal = f"""
---

## 🔋 Battery Trading Analysis

### Battery Specifications
- **Capacity**: {self.specs.max_charge_mw:.0f} MW / {self.specs.capacity_mwh:.0f} MWh
- **Round-trip Efficiency**: {self.specs.round_trip_efficiency*100:.1f}%
- **SOC Operating Range**: {self.specs.min_soc*100:.0f}% - {self.specs.max_soc*100:.0f}%

### Market Conditions
- **Mean LMP**: ${lmp_mean:.2f}/MWh
- **LMP Volatility**: ${lmp_std:.2f}/MWh (CoV: {lmp_std/lmp_mean*100:.1f}%)
- **Price Range**: ${lmp_min:.2f} - ${lmp_max:.2f}/MWh
- **Observations**: {len(lmp_data):,} intervals

### State of Charge (SOC) Performance
- **Mean SOC**: {metrics['soc_mean']*100:.1f}%
- **SOC Range**: {metrics['soc_min']*100:.1f}% - {metrics['soc_max']*100:.1f}%
- **SOC Utilization**: {metrics['soc_utilization_pct']:.1f}% of available capacity
- **Standard Deviation**: {metrics['soc_std']*100:.1f}%

### Operational Metrics
- **Charging Intervals**: {metrics['charge_intervals']} ({metrics['charge_intervals']/len(results['action'])*100:.1f}%)
- **Discharging Intervals**: {metrics['discharge_intervals']} ({metrics['discharge_intervals']/len(results['action'])*100:.1f}%)
- **Idle Intervals**: {metrics['idle_intervals']} ({metrics['idle_intervals']/len(results['action'])*100:.1f}%)
- **Active Utilization**: {metrics['active_pct']:.1f}%
- **Estimated Cycles**: {metrics['estimated_cycles']:.2f}

### Power Performance
- **Avg Charging Power**: {abs(metrics['avg_charge_power_mw']):.1f} MW
- **Avg Discharging Power**: {metrics['avg_discharge_power_mw']:.1f} MW
- **Peak Power**: {metrics['max_power_mw']:.1f} MW

### Revenue & Profitability
- **Discharge Revenue**: ${metrics['discharge_revenue']:.2f}
- **Charging Cost**: ${abs(metrics['charge_cost']):.2f}
- **Net Profit/Loss**: ${metrics['final_pnl']:.2f}
- **Gross Margin**: {metrics['gross_profit_margin']:.1f}%
- **Avg Revenue/Interval**: ${metrics['avg_revenue_per_interval']:.2f}

### Trading Strategy Performance
- **Strategy**: Simple Arbitrage (charge low, discharge high)
- **Market Design**: Current ERCOT SCED
- **Profit per Cycle**: ${metrics['final_pnl']/max(metrics['estimated_cycles'], 0.1):.2f}
- **Efficiency vs Theory**: {metrics['gross_profit_margin']/100:.1%} (vs {self.specs.round_trip_efficiency:.1%} theoretical)

---

## 📝 Battery Research Observations

### Key Findings Today
- [ ] **Price Patterns**: {f"High volatility ({lmp_std:.1f})" if lmp_std > 20 else f"Low volatility ({lmp_std:.1f})"} - impact on arbitrage opportunities
- [ ] **SOC Management**: {'Underutilized' if metrics['soc_utilization_pct'] < 50 else 'Well-utilized'} ({metrics['soc_utilization_pct']:.0f}% of capacity)
- [ ] **Cycling**: {metrics['estimated_cycles']:.2f} cycles - {'within' if metrics['estimated_cycles'] < 2 else 'exceeds'} daily target
- [ ] **Profitability**: {'Profitable' if metrics['final_pnl'] > 0 else 'Loss'} trading day (${metrics['final_pnl']:.2f})

### Optimal Trading Times (Observations)
- **Best discharge opportunities**: {f"Prices > ${lmp_mean + lmp_std:.0f}/MWh" if lmp_std > 0 else "N/A"}
- **Best charge opportunities**: {f"Prices < ${lmp_mean - lmp_std:.0f}/MWh" if lmp_std > 0 else "N/A"}
- **Price spread captured**: ${lmp_max - lmp_min:.2f}/MWh

### Strategy Performance Notes
- [ ] *Did the strategy capture major price movements?*
- [ ] *Were there missed opportunities (idle during high spreads)?*
- [ ] *How did forecast accuracy affect bidding decisions?*

### Comparison with RTC+B (Future)
- [ ] *How might co-optimization change bidding strategy?*
- [ ] *Would ancillary service participation be more profitable?*
- [ ] *Impact of ASDCs on revenue potential?*

### Questions for Further Investigation
- [ ] *What price volatility threshold makes arbitrage most profitable?*
- [ ] *Optimal SOC management strategy for this price pattern?*
- [ ] *Impact of cycling frequency on degradation costs?*

---
"""
        return journal


def run_battery_daily_analysis(lmp_data: pd.DataFrame) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Run complete battery daily analysis.
    
    Parameters
    ----------
    lmp_data : pd.DataFrame
        LMP data with timestamp, lmp_usd, settlement_point columns
        
    Returns
    -------
    tuple of (results, metrics, journal_section)
        results : dict - Simulation results
        metrics : dict - Performance metrics
        journal_section : str - Markdown journal text
    """
    analyzer = BatteryDailyAnalyzer()
    
    # Run simulation
    results = analyzer.run_arbitrage_simulation(lmp_data, market_design=MarketDesign.SCED)
    
    # Compute metrics
    metrics = analyzer.compute_battery_metrics(results)
    
    # Generate journal section
    journal_section = analyzer.generate_battery_journal_section(lmp_data, results, metrics)
    
    return results, metrics, journal_section
