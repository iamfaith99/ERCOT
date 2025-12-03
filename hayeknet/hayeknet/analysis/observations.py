"""Research observations tracking system.

Generates structured observation files for graduate research,
tracking patterns, hypotheses, and findings over time.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd


class ResearchObservationTracker:
    """Track and generate research observations across multiple days."""
    
    def __init__(self, observations_dir: Path):
        """Initialize observation tracker.
        
        Parameters
        ----------
        observations_dir : Path
            Directory to store observation files
        """
        self.observations_dir = Path(observations_dir)
        self.observations_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_daily_observation_template(
        self,
        date: datetime,
        market_data: pd.DataFrame,
        system_results: Dict[str, Any],
        battery_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate daily observation template with auto-filled data and prompts.
        
        Parameters
        ----------
        date : datetime
            Date of observations
        market_data : pd.DataFrame
            Market data for the day
        system_results : dict
            HayekNet system results
        battery_metrics : dict
            Battery performance metrics
            
        Returns
        -------
        str
            Markdown-formatted observation template
        """
        day_num = (date - datetime(2025, 10, 1)).days + 1
        days_remaining = (datetime(2025, 12, 5) - date).days
        
        # Market analysis
        if not market_data.empty and 'lmp_usd' in market_data.columns:
            lmp_mean = market_data['lmp_usd'].mean()
            lmp_std = market_data['lmp_usd'].std()
            lmp_cov = (lmp_std / lmp_mean * 100) if lmp_mean > 0 else 0
        else:
            lmp_mean = lmp_std = lmp_cov = 0
        
        # Battery analysis
        battery_pnl = battery_metrics.get('final_pnl', 0)
        battery_profitable = battery_pnl > 0
        soc_util = battery_metrics.get('soc_utilization_pct', 0)
        cycles = battery_metrics.get('estimated_cycles', 0)
        
        template = f"""# Daily Research Observation
**Date**: {date.strftime('%Y-%m-%d (%A)')}  
**Day**: {day_num} of research period  
**Days Remaining**: {days_remaining} until Dec 5, 2025  

---

## 📊 Market Characterization

### Today's Market Type
- [ ] **Low Volatility** (CoV < 10%)
- [{'x' if lmp_cov < 10 else ' '}] **Moderate Volatility** (CoV 10-25%)
- [ ] **High Volatility** (CoV > 25%)
- [ ] **Scarcity Event** (prices > $100/MWh sustained)
- [ ] **Normal Operations** (typical demand pattern)

**Actual**: CoV = {lmp_cov:.1f}%, Mean LMP = ${lmp_mean:.2f}/MWh

### Price Patterns Observed
- [ ] Flat pricing (minimal intraday variation)
- [ ] Clear peak/off-peak structure
- [ ] Multiple price spikes
- [ ] Sustained high prices (>$100/MWh)
- [ ] Negative or near-zero prices
- [ ] Congestion-driven price separation

**Notes**: *Describe what you observed...*

---

## 🔋 Battery Performance Analysis

### Profitability Assessment
**Result**: {'✅ PROFITABLE' if battery_profitable else '❌ UNPROFITABLE'} (PnL: ${battery_pnl:.2f})

### Why was today profitable/unprofitable?
- [ ] Sufficient price spreads (>$20/MWh spread)
- [ ] Insufficient price spreads (<$10/MWh spread)
- [ ] Good timing (charged at lows, discharged at peaks)
- [ ] Poor timing (charged when prices were rising)
- [ ] Low SOC utilization ({soc_util:.1f}% - underutilized)
- [ ] High SOC utilization ({soc_util:.1f}% - well-utilized)
- [ ] Strategy too conservative
- [ ] Strategy too aggressive

**Your Analysis**: *What was the key factor?*

### SOC Management
- **Utilization**: {soc_util:.1f}%
- **Cycles**: {cycles:.2f}

**Observations**:
- [ ] Battery stayed in middle SOC range (50-60%)
- [ ] Battery cycled fully (used 70-80% capacity)
- [ ] Battery hit SOC limits (10% or 90%)
- [ ] Optimal SOC management observed
- [ ] Suboptimal SOC management observed

**Notes**: *How should SOC be managed for this market type?*

---

## 🎯 Strategy Performance

### Simple Arbitrage Strategy
- **Charge threshold**: 30th percentile
- **Discharge threshold**: 70th percentile

**Effectiveness Today**:
- [ ] Excellent - Captured all major opportunities
- [ ] Good - Captured most opportunities
- [ ] Fair - Missed some opportunities
- [ ] Poor - Missed most opportunities

**Missed Opportunities**:
- [ ] Yes - __ intervals where spread was profitable but battery was idle
- [ ] No - Strategy executed as expected

**Notes**: *What could be improved?*

### Comparison: What if we had...
- [ ] Different charge/discharge thresholds?
- [ ] Participated in ancillary services?
- [ ] Had RTC+B co-optimization?
- [ ] Used forecast-based strategy instead of reactive?

**Thoughts**: *Potential improvements...*

---

## 🔬 Research Hypotheses

### Hypothesis Testing

**H1: Higher price volatility → Higher arbitrage profitability**
- Market volatility today: {lmp_cov:.1f}%
- Battery PnL: ${battery_pnl:.2f}
- [{'✅' if lmp_cov > 15 and battery_profitable else '❓'}] Supports hypothesis
- **Notes**: *Update running correlation analysis*

**H2: Optimal SOC utilization is 60-80% for maximizing cycles**
- Utilization today: {soc_util:.1f}%
- Cycles: {cycles:.2f}
- [{'✅' if 60 <= soc_util <= 80 else '❓'}] Within optimal range
- **Notes**: *Track relationship*

**H3: Peak/off-peak spread must be >$15/MWh for profitability**
- Price spread today: ${battery_metrics.get('discharge_revenue', 0) / max(battery_metrics.get('discharge_intervals', 1), 1) - abs(battery_metrics.get('charge_cost', 0)) / max(battery_metrics.get('charge_intervals', 1), 1):.2f}/MWh
- Profitable: {battery_profitable}
- [{'✅' if abs(battery_pnl) > 500 and battery_profitable else '❓'}] Supports hypothesis
- **Notes**: *Threshold verification*

### New Hypotheses Generated Today
- [ ] *Add new hypothesis based on today's observations...*

---

## 📈 Patterns & Trends

### Multi-Day Patterns (Update Weekly)
- [ ] Weekday vs weekend differences
- [ ] Time-of-day patterns
- [ ] Weather correlation
- [ ] Seasonal trends (as data accumulates)

**Notes**: *Patterns emerging over past week...*

### Accumulating Evidence For
- [ ] **Finding**: *State an emerging finding...*
  - Evidence: Day {day_num} observation supports this
  - Confidence: Low / Medium / High
  - Days observed: __ / {day_num}

---

## 💡 Insights for Thesis

### Chapter 3 (Methodology)
- [ ] Validated battery model with real data
- [ ] Identified strategy parameter sensitivity
- [ ] Documented market conditions tested

**Notes**: *Method refinements needed...*

### Chapter 4 (Results)
**Key Finding Today**: *One-sentence summary of most important result*

**Quantitative Evidence**:
- PnL: ${battery_pnl:.2f}
- Volatility: {lmp_cov:.1f}%
- Utilization: {soc_util:.1f}%

**Figures Needed**: 
- [ ] SOC trajectory for today
- [ ] Price vs battery actions
- [ ] Cumulative PnL over time

### Chapter 5 (Discussion)
**Implication**: *What does today's result mean for the research question?*

**Connection to Literature**:
- [ ] Relates to [Author, Year]: *How?*
- [ ] Challenges/Supports [Author, Year]: *How?*

---

## 🔄 RTC+B Considerations

### How might RTC+B change today's results?

**Ancillary Service Opportunities**:
- [ ] RegUp prices likely high (volatile market)
- [ ] RegDown prices likely high
- [ ] RRS prices elevated
- [ ] Co-optimization would likely increase revenue by: *estimate $X*

**Strategy Changes Under RTC+B**:
- [ ] Would participate in AS instead of pure arbitrage
- [ ] Would maintain headroom for AS commitments
- [ ] Would bid differently based on ASDCs
- [ ] Would see XX% revenue improvement (estimate)

**Notes**: *RTC+B comparison thoughts...*

---

## ❓ Questions Raised Today

### Technical Questions
1. *Question about system performance?*
   - [ ] Investigate further
   - [ ] Answered: *Answer here*

### Strategy Questions
1. *Question about trading strategy?*
   - [ ] Test hypothesis
   - [ ] Answered: *Answer here*

### Research Questions
1. *Question for thesis investigation?*
   - [ ] Add to research plan
   - [ ] Answered: *Answer here*

---

## 📋 Action Items

### For Tomorrow's Run
- [ ] Monitor [specific metric]
- [ ] Test [parameter change]
- [ ] Verify [hypothesis]

### For Week Summary
- [ ] Compile [data]
- [ ] Analyze [pattern]
- [ ] Generate [figure]

### For Advisor Meeting
- [ ] Discuss [finding]
- [ ] Get feedback on [approach]
- [ ] Clarify [question]

---

## 🔖 Tags & Keywords
`day-{day_num}` `market-{'volatile' if lmp_cov > 15 else 'stable'}` `battery-{'profitable' if battery_profitable else 'unprofitable'}` `soc-{'high' if soc_util > 50 else 'low'}-util`

---

*Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Remember to fill in your observations and analysis!*
"""
        
        return template
    
    def save_daily_observation(
        self,
        date: datetime,
        observation_content: str
    ) -> Path:
        """
        Save daily observation to file.
        
        Parameters
        ----------
        date : datetime
            Date of observation
        observation_content : str
            Observation markdown content
            
        Returns
        -------
        Path
            Path to saved observation file
        """
        filename = f"observation_{date.strftime('%Y-%m-%d')}.md"
        filepath = self.observations_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(observation_content)
        
        return filepath
    
    def generate_weekly_summary(self, end_date: datetime) -> str:
        """
        Generate weekly summary of observations.
        
        Parameters
        ----------
        end_date : datetime
            End date of week
            
        Returns
        -------
        str
            Weekly summary markdown
        """
        start_date = end_date - timedelta(days=6)
        
        # Find observation files for this week
        obs_files = sorted(self.observations_dir.glob('observation_*.md'))
        week_files = [
            f for f in obs_files
            if start_date.strftime('%Y-%m-%d') <= f.stem.replace('observation_', '') <= end_date.strftime('%Y-%m-%d')
        ]
        
        summary = f"""# Weekly Research Summary
**Week Ending**: {end_date.strftime('%Y-%m-%d')}  
**Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}  
**Days Observed**: {len(week_files)}

---

## 📊 Week Overview

### Market Conditions This Week
- [ ] Mostly low volatility days
- [ ] Mixed volatility
- [ ] Mostly high volatility days
- [ ] Scarcity event(s) occurred

**Summary**: *Characterize the week's market conditions...*

### Battery Performance Summary
- **Days Profitable**: __ / {len(week_files)}
- **Total PnL**: $____ 
- **Best Day**: {end_date.strftime('%Y-%m-%d')} ($____)
- **Worst Day**: {end_date.strftime('%Y-%m-%d')} ($____)

**Key Insight**: *What determined profitability this week?*

---

## 🔬 Hypothesis Status

### H1: Volatility → Profitability
- **Evidence This Week**: 
- **Confidence**: Low / Medium / High
- **Status**: Supported / Refuted / Unclear

### H2: SOC Utilization 60-80%
- **Evidence This Week**:
- **Confidence**: Low / Medium / High
- **Status**: Supported / Refuted / Unclear

### H3: Spread >$15/MWh Required
- **Evidence This Week**:
- **Confidence**: Low / Medium / High  
- **Status**: Supported / Refuted / Unclear

---

## 📈 Patterns Identified

### Pattern 1: [Name]
- **Observed Days**: __ / {len(week_files)}
- **Description**: *What pattern emerged?*
- **Implication**: *What does it mean?*

### Pattern 2: [Name]
- **Observed Days**: __ / {len(week_files)}
- **Description**: *What pattern emerged?*
- **Implication**: *What does it mean?*

---

## 💡 Insights for Thesis

### Most Important Finding This Week
*One paragraph summary of the key insight...*

### Evidence Collected
- Quantitative: *Data points*
- Qualitative: *Observations*

### Figures to Generate
- [ ] Figure: *Description*
- [ ] Figure: *Description*

---

## 🔄 Strategy Adjustments Needed

Based on this week's observations:
- [ ] Adjust charge threshold to __
- [ ] Adjust discharge threshold to __
- [ ] Consider AS participation
- [ ] Test different SOC management

---

## 📋 Next Week Focus

### Research Questions
1. *Question to investigate...*
2. *Question to investigate...*

### Data to Collect
- [ ] *Specific metric*
- [ ] *Specific condition*

### Analysis to Perform
- [ ] *Analysis type*
- [ ] *Comparison*

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary
    
    def save_weekly_summary(self, end_date: datetime, summary_content: str) -> Path:
        """Save weekly summary."""
        filename = f"week_ending_{end_date.strftime('%Y-%m-%d')}.md"
        filepath = self.observations_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(summary_content)
        
        return filepath
