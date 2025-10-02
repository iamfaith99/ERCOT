"""Automated Research Analysis System for HayekNet.

This module provides AI-powered analysis capabilities that:
1. Analyze daily results and generate insights
2. Ask and answer research questions automatically  
3. Identify patterns and anomalies
4. Generate hypothesis tests and statistical analysis
5. Create narrative summaries of findings

The system acts as an AI research assistant, filling in analytical 
gaps and providing continuous insight generation.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ResearchInsight:
    """Single research insight or finding."""
    date: datetime
    insight_type: str  # "pattern", "anomaly", "hypothesis", "trend", "question"
    title: str
    description: str
    evidence: Dict[str, Any]
    confidence: float  # 0-1
    significance: str  # "low", "medium", "high"
    follow_up_questions: List[str]
    statistical_support: Optional[Dict[str, float]] = None

class AutomatedResearchAnalyst:
    """AI-powered research analysis system for HayekNet."""
    
    def __init__(self, research_dir: Path):
        """Initialize the automated research analyst.
        
        Parameters
        ----------
        research_dir : Path
            Directory containing research outputs
        """
        self.research_dir = Path(research_dir)
        self.insights_dir = self.research_dir / "insights"
        self.insights_dir.mkdir(parents=True, exist_ok=True)
        
        # Historical insights for pattern detection
        self.insight_history = []
        self.load_historical_insights()
        
    def analyze_daily_results(
        self,
        date: datetime,
        market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any],
        agent_performance: Dict[str, Any],
        system_results: Dict[str, Any]
    ) -> List[ResearchInsight]:
        """Comprehensive analysis of daily results with automated insights.
        
        Parameters
        ----------
        date : datetime
            Date of analysis
        market_data : pd.DataFrame
            ERCOT market data
        battery_metrics : dict
            Battery performance metrics
        agent_performance : dict
            RL agent performance data
        system_results : dict
            Overall system results
            
        Returns
        -------
        insights : List[ResearchInsight]
            Generated research insights
        """
        insights = []
        
        print(f"🤖 Running automated analysis for {date.strftime('%Y-%m-%d')}...")
        
        # Market Analysis
        market_insights = self._analyze_market_patterns(date, market_data)
        insights.extend(market_insights)
        
        # Battery Performance Analysis  
        battery_insights = self._analyze_battery_performance(date, battery_metrics, market_data)
        insights.extend(battery_insights)
        
        # Agent Learning Analysis
        agent_insights = self._analyze_agent_learning(date, agent_performance)
        insights.extend(agent_insights)
        
        # Cross-System Analysis
        system_insights = self._analyze_system_interactions(
            date, market_data, battery_metrics, agent_performance
        )
        insights.extend(system_insights)
        
        # Multi-Day Trend Analysis
        if len(self.insight_history) > 5:  # Need history for trends
            trend_insights = self._analyze_trends(date, insights)
            insights.extend(trend_insights)
        
        # Generate Follow-up Questions
        self._generate_research_questions(insights)
        
        # Save insights
        self._save_insights(date, insights)
        self.insight_history.extend(insights)
        
        print(f"✅ Generated {len(insights)} research insights")
        return insights
    
    def _analyze_market_patterns(self, date: datetime, market_data: pd.DataFrame) -> List[ResearchInsight]:
        """Analyze market data patterns and generate insights."""
        insights = []
        
        if market_data.empty or 'lmp_usd' not in market_data.columns:
            return insights
        
        lmp = market_data['lmp_usd']
        
        # Price Statistics
        lmp_mean = lmp.mean()
        lmp_std = lmp.std()
        lmp_cov = (lmp_std / lmp_mean * 100) if lmp_mean > 0 else 0
        lmp_min, lmp_max = lmp.min(), lmp.max()
        price_spread = lmp_max - lmp_min
        
        # Detect Market Regime
        if lmp_cov > 30:
            market_type = "High Volatility"
            significance = "high"
        elif lmp_cov < 10:
            market_type = "Low Volatility"  
            significance = "medium"
        else:
            market_type = "Normal Volatility"
            significance = "low"
            
        insights.append(ResearchInsight(
            date=date,
            insight_type="pattern",
            title=f"Market Regime: {market_type}",
            description=f"Today's market showed {market_type.lower()} with CoV={lmp_cov:.1f}%. "
                       f"Price range: ${lmp_min:.2f}-${lmp_max:.2f}/MWh (spread: ${price_spread:.2f}).",
            evidence={
                "coefficient_of_variation": lmp_cov,
                "mean_lmp": lmp_mean,
                "std_lmp": lmp_std,
                "price_spread": price_spread,
                "regime_classification": market_type
            },
            confidence=0.9,
            significance=significance,
            follow_up_questions=[
                f"What caused the {market_type.lower()} today?",
                "How does this compare to historical patterns?",
                "What are the implications for battery strategy?"
            ]
        ))
        
        # Detect Price Spikes (>$100/MWh)
        spikes = lmp[lmp > 100]
        if len(spikes) > 0:
            spike_duration = len(spikes) * 5  # 5-minute intervals
            insights.append(ResearchInsight(
                date=date,
                insight_type="anomaly",
                title="Price Spike Event",
                description=f"Detected {len(spikes)} price intervals >$100/MWh, "
                           f"lasting {spike_duration} minutes. Peak: ${spikes.max():.2f}/MWh.",
                evidence={
                    "spike_count": len(spikes),
                    "spike_duration_minutes": spike_duration,
                    "max_spike_price": spikes.max(),
                    "spike_percentage": len(spikes) / len(lmp) * 100
                },
                confidence=0.95,
                significance="high",
                follow_up_questions=[
                    "What grid conditions caused this spike?",
                    "Did our battery capitalize on this opportunity?", 
                    "How can we better predict such events?"
                ]
            ))
        
        # Intraday Pattern Analysis
        if len(market_data) >= 24:  # At least 24 intervals (2 hours)
            market_data_copy = market_data.copy()
            market_data_copy['hour'] = pd.to_datetime(market_data_copy.get('timestamp', market_data_copy.index)).dt.hour
            
            if 'hour' in market_data_copy.columns:
                hourly_avg = market_data_copy.groupby('hour')['lmp_usd'].mean()
                peak_hour = hourly_avg.idxmax()
                off_peak_hour = hourly_avg.idxmin()
                peak_offpeak_spread = hourly_avg.max() - hourly_avg.min()
                
                if peak_offpeak_spread > 20:
                    insights.append(ResearchInsight(
                        date=date,
                        insight_type="pattern",
                        title="Strong Intraday Pattern",
                        description=f"Clear peak/off-peak pattern observed. Peak at hour {peak_hour} "
                                   f"(${hourly_avg.max():.2f}/MWh), valley at hour {off_peak_hour} "
                                   f"(${hourly_avg.min():.2f}/MWh). Spread: ${peak_offpeak_spread:.2f}/MWh.",
                        evidence={
                            "peak_hour": peak_hour,
                            "off_peak_hour": off_peak_hour,
                            "peak_price": hourly_avg.max(),
                            "off_peak_price": hourly_avg.min(),
                            "peak_offpeak_spread": peak_offpeak_spread
                        },
                        confidence=0.85,
                        significance="high" if peak_offpeak_spread > 30 else "medium",
                        follow_up_questions=[
                            "Is this pattern consistent across weekdays/weekends?",
                            "How well did our charging strategy align with this pattern?",
                            "Could we improve timing based on this pattern?"
                        ]
                    ))
        
        return insights
    
    def _analyze_battery_performance(
        self, 
        date: datetime, 
        battery_metrics: Dict[str, Any], 
        market_data: pd.DataFrame
    ) -> List[ResearchInsight]:
        """Analyze battery performance and generate insights."""
        insights = []
        
        pnl = battery_metrics.get('final_pnl', 0)
        soc_utilization = battery_metrics.get('soc_utilization_pct', 0)
        cycles = battery_metrics.get('estimated_cycles', 0)
        charge_intervals = battery_metrics.get('charge_intervals', 0)
        discharge_intervals = battery_metrics.get('discharge_intervals', 0)
        
        # Profitability Analysis
        profitable = pnl > 0
        profitability_level = (
            "highly profitable" if pnl > 1000 else
            "moderately profitable" if pnl > 100 else  
            "marginally profitable" if pnl > 0 else
            "marginally unprofitable" if pnl > -100 else
            "significantly unprofitable"
        )
        
        insights.append(ResearchInsight(
            date=date,
            insight_type="performance",
            title=f"Battery Performance: {profitability_level.title()}",
            description=f"Battery achieved ${pnl:.2f} PnL with {soc_utilization:.1f}% SOC utilization "
                       f"and {cycles:.2f} equivalent cycles. Strategy was {profitability_level}.",
            evidence={
                "pnl": pnl,
                "profitable": profitable,
                "soc_utilization": soc_utilization,
                "cycles": cycles,
                "charge_intervals": charge_intervals,
                "discharge_intervals": discharge_intervals,
                "profitability_category": profitability_level
            },
            confidence=0.95,
            significance="high" if abs(pnl) > 500 else "medium",
            follow_up_questions=[
                f"What market conditions led to this {profitability_level} outcome?",
                f"Is {soc_utilization:.1f}% SOC utilization optimal?",
                "How does this compare to our theoretical maximum?"
            ]
        ))
        
        # SOC Management Analysis
        if soc_utilization < 30:
            insights.append(ResearchInsight(
                date=date,
                insight_type="pattern",
                title="Low SOC Utilization",
                description=f"Battery only used {soc_utilization:.1f}% of available capacity. "
                           f"This suggests either conservative strategy or limited opportunities.",
                evidence={
                    "soc_utilization": soc_utilization,
                    "charge_intervals": charge_intervals,
                    "discharge_intervals": discharge_intervals
                },
                confidence=0.8,
                significance="medium",
                follow_up_questions=[
                    "Were there more arbitrage opportunities we missed?",
                    "Should we adjust charge/discharge thresholds?",
                    "What's the optimal utilization for this market type?"
                ]
            ))
        elif soc_utilization > 80:
            insights.append(ResearchInsight(
                date=date,
                insight_type="pattern", 
                title="High SOC Utilization",
                description=f"Battery used {soc_utilization:.1f}% of capacity - very active trading. "
                           f"This indicates either good opportunities or aggressive strategy.",
                evidence={
                    "soc_utilization": soc_utilization,
                    "cycles": cycles
                },
                confidence=0.8,
                significance="medium",
                follow_up_questions=[
                    "Was this high utilization profitable?",
                    "Are we optimizing for revenue or battery life?",
                    "What's the degradation cost vs profit tradeoff?"
                ]
            ))
            
        return insights
    
    def _analyze_agent_learning(self, date: datetime, agent_performance: Dict[str, Any]) -> List[ResearchInsight]:
        """Analyze RL agent learning progress and generate insights.""" 
        insights = []
        
        if not agent_performance:
            return insights
            
        # Training Progress
        training_timesteps = agent_performance.get('total_timesteps', 0)
        mean_reward = agent_performance.get('mean_reward', 0)
        reward_std = agent_performance.get('reward_std', 0)
        
        if training_timesteps > 0:
            learning_stage = (
                "early learning" if training_timesteps < 50_000 else
                "intermediate learning" if training_timesteps < 200_000 else 
                "advanced learning"
            )
            
            insights.append(ResearchInsight(
                date=date,
                insight_type="trend",
                title=f"Agent Learning Progress: {learning_stage.title()}",
                description=f"RL agents have completed {training_timesteps:,} training steps "
                           f"with mean reward {mean_reward:.2f} ± {reward_std:.2f}. "
                           f"Agents are in {learning_stage} phase.",
                evidence={
                    "training_timesteps": training_timesteps,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "learning_stage": learning_stage
                },
                confidence=0.75,
                significance="medium" if training_timesteps > 100_000 else "low",
                follow_up_questions=[
                    "Is the reward signal improving over time?",
                    "Are agents converging to optimal strategies?",
                    "How does performance compare to simple heuristics?"
                ]
            ))
        
        return insights
    
    def _analyze_system_interactions(
        self, 
        date: datetime,
        market_data: pd.DataFrame, 
        battery_metrics: Dict[str, Any],
        agent_performance: Dict[str, Any]
    ) -> List[ResearchInsight]:
        """Analyze interactions between market, battery, and agents."""
        insights = []
        
        if market_data.empty or 'lmp_usd' not in market_data.columns:
            return insights
            
        # Market-Battery Performance Correlation
        lmp_volatility = market_data['lmp_usd'].std()
        battery_pnl = battery_metrics.get('final_pnl', 0)
        
        # Simple correlation analysis
        if lmp_volatility > 20 and battery_pnl > 100:
            insights.append(ResearchInsight(
                date=date,
                insight_type="hypothesis",
                title="Volatility-Profitability Correlation",
                description=f"High market volatility (σ=${lmp_volatility:.2f}) coincided with "
                           f"profitable battery operation (${battery_pnl:.2f}). "
                           f"This supports the volatility-profitability hypothesis.",
                evidence={
                    "market_volatility": lmp_volatility,
                    "battery_pnl": battery_pnl,
                    "supports_hypothesis": True
                },
                confidence=0.7,
                significance="medium",
                follow_up_questions=[
                    "What's the threshold volatility for profitability?",
                    "Is this correlation consistent across different market conditions?",
                    "How can we better exploit high-volatility periods?"
                ]
            ))
        elif lmp_volatility < 10 and battery_pnl < 0:
            insights.append(ResearchInsight(
                date=date,
                insight_type="hypothesis",
                title="Low Volatility = Low Profitability",
                description=f"Low market volatility (σ=${lmp_volatility:.2f}) led to "
                           f"unprofitable battery operation (${battery_pnl:.2f}). "
                           f"This confirms that arbitrage requires price volatility.",
                evidence={
                    "market_volatility": lmp_volatility,
                    "battery_pnl": battery_pnl,
                    "supports_low_vol_hypothesis": True
                },
                confidence=0.8,
                significance="medium",
                follow_up_questions=[
                    "Should we avoid trading on low-volatility days?",
                    "Are there other revenue streams (AS) we should pursue?",
                    "What's our minimum volatility threshold?"
                ]
            ))
        
        return insights
    
    def _analyze_trends(self, date: datetime, current_insights: List[ResearchInsight]) -> List[ResearchInsight]:
        """Analyze multi-day trends and patterns."""
        insights = []
        
        # Look at last 7 days of insights
        recent_insights = [
            insight for insight in self.insight_history 
            if (date - insight.date).days <= 7
        ]
        
        if len(recent_insights) < 3:
            return insights
            
        # Trend in profitability
        profitable_days = sum(1 for insight in recent_insights 
                            if insight.insight_type == "performance" and 
                            insight.evidence.get('profitable', False))
        
        total_performance_insights = sum(1 for insight in recent_insights 
                                       if insight.insight_type == "performance")
        
        if total_performance_insights >= 3:
            profit_rate = profitable_days / total_performance_insights
            
            if profit_rate >= 0.7:
                trend_desc = "consistently profitable"
            elif profit_rate >= 0.4:
                trend_desc = "mixed profitability"
            else:
                trend_desc = "consistently unprofitable"
                
            insights.append(ResearchInsight(
                date=date,
                insight_type="trend",
                title=f"Weekly Trend: {trend_desc.title()}",
                description=f"Over the past week, battery was profitable {profitable_days}/"
                           f"{total_performance_insights} days ({profit_rate*100:.0f}%). "
                           f"This indicates {trend_desc} performance.",
                evidence={
                    "profitable_days": profitable_days,
                    "total_days": total_performance_insights,
                    "profit_rate": profit_rate,
                    "trend_classification": trend_desc
                },
                confidence=0.8,
                significance="high" if total_performance_insights >= 5 else "medium",
                follow_up_questions=[
                    "What market conditions drive profitable vs unprofitable days?",
                    "Is our strategy improving over time?",
                    "Should we adjust strategy based on this trend?"
                ]
            ))
        
        return insights
    
    def _generate_research_questions(self, insights: List[ResearchInsight]):
        """Generate follow-up research questions based on insights."""
        all_questions = []
        for insight in insights:
            all_questions.extend(insight.follow_up_questions)
        
        # Deduplicate and prioritize questions
        unique_questions = list(set(all_questions))
        
        # Add to each insight (for now, just store them)
        for insight in insights:
            if not insight.follow_up_questions:
                insight.follow_up_questions = unique_questions[:3]  # Top 3
    
    def generate_narrative_summary(self, date: datetime, insights: List[ResearchInsight]) -> str:
        """Generate narrative summary of the day's findings."""
        if not insights:
            return f"No significant insights generated for {date.strftime('%Y-%m-%d')}."
        
        # Categorize insights
        high_significance = [i for i in insights if i.significance == "high"]
        patterns = [i for i in insights if i.insight_type == "pattern"]
        anomalies = [i for i in insights if i.insight_type == "anomaly"]
        performance = [i for i in insights if i.insight_type == "performance"]
        
        narrative = f"# AI Analysis Summary - {date.strftime('%Y-%m-%d')}\n\n"
        
        # Key Findings
        if high_significance:
            narrative += "## 🔍 Key Findings\n\n"
            for insight in high_significance[:3]:  # Top 3 most significant
                narrative += f"**{insight.title}**: {insight.description}\n\n"
        
        # Market Analysis
        if patterns:
            narrative += "## 📊 Market Analysis\n\n"
            for insight in patterns:
                if 'market' in insight.title.lower() or 'price' in insight.title.lower():
                    narrative += f"- **{insight.title}**: {insight.description}\n"
            narrative += "\n"
        
        # Performance Summary
        if performance:
            narrative += "## 🔋 Performance Analysis\n\n"
            for insight in performance:
                narrative += f"- **{insight.title}**: {insight.description}\n"
            narrative += "\n"
        
        # Anomalies & Alerts
        if anomalies:
            narrative += "## ⚠️ Anomalies Detected\n\n"
            for insight in anomalies:
                narrative += f"- **{insight.title}**: {insight.description}\n"
            narrative += "\n"
        
        # Research Questions
        all_questions = []
        for insight in insights:
            all_questions.extend(insight.follow_up_questions)
        unique_questions = list(set(all_questions))
        
        if unique_questions:
            narrative += "## ❓ Research Questions Generated\n\n"
            for i, question in enumerate(unique_questions[:5], 1):  # Top 5
                narrative += f"{i}. {question}\n"
            narrative += "\n"
        
        # Confidence Assessment
        high_conf_insights = [i for i in insights if i.confidence >= 0.8]
        narrative += f"## 📈 Analysis Quality\n\n"
        narrative += f"- **Total Insights**: {len(insights)}\n"
        narrative += f"- **High Confidence**: {len(high_conf_insights)} ({len(high_conf_insights)/len(insights)*100:.0f}%)\n"
        narrative += f"- **High Significance**: {len(high_significance)} ({len(high_significance)/len(insights)*100:.0f}%)\n\n"
        
        narrative += "---\n*Generated by HayekNet Automated Research Analyst*"
        
        return narrative
    
    def _save_insights(self, date: datetime, insights: List[ResearchInsight]):
        """Save insights to structured files."""
        date_str = date.strftime('%Y-%m-%d')
        
        # Save raw insights as JSON
        insights_data = []
        for insight in insights:
            insight_dict = {
                'date': insight.date.isoformat(),
                'insight_type': insight.insight_type,
                'title': insight.title,
                'description': insight.description,
                'evidence': insight.evidence,
                'confidence': insight.confidence,
                'significance': insight.significance,
                'follow_up_questions': insight.follow_up_questions,
                'statistical_support': insight.statistical_support
            }
            insights_data.append(insight_dict)
        
        insights_file = self.insights_dir / f"insights_{date_str}.json"
        with open(insights_file, 'w') as f:
            json.dump(insights_data, f, indent=2)
        
        # Save narrative summary
        narrative = self.generate_narrative_summary(date, insights)
        summary_file = self.insights_dir / f"analysis_summary_{date_str}.md"
        with open(summary_file, 'w') as f:
            f.write(narrative)
        
        print(f"💾 Saved insights to {insights_file}")
        print(f"📝 Saved summary to {summary_file}")
    
    def load_historical_insights(self):
        """Load previously generated insights for trend analysis."""
        insight_files = list(self.insights_dir.glob("insights_*.json"))
        
        for file in sorted(insight_files):
            try:
                with open(file, 'r') as f:
                    insights_data = json.load(f)
                
                for data in insights_data:
                    insight = ResearchInsight(
                        date=datetime.fromisoformat(data['date']),
                        insight_type=data['insight_type'],
                        title=data['title'],
                        description=data['description'],
                        evidence=data['evidence'],
                        confidence=data['confidence'],
                        significance=data['significance'],
                        follow_up_questions=data['follow_up_questions'],
                        statistical_support=data.get('statistical_support')
                    )
                    self.insight_history.append(insight)
                    
            except Exception as e:
                print(f"Warning: Could not load insights from {file}: {e}")
        
        print(f"📚 Loaded {len(self.insight_history)} historical insights")


def run_automated_analysis(
    research_dir: Path,
    date: datetime,
    market_data: pd.DataFrame,
    battery_metrics: Dict[str, Any],
    agent_performance: Optional[Dict[str, Any]] = None,
    system_results: Optional[Dict[str, Any]] = None
) -> List[ResearchInsight]:
    """Run automated analysis and generate insights.
    
    Convenience function for running the full analysis pipeline.
    
    Parameters
    ----------
    research_dir : Path
        Research directory
    date : datetime
        Analysis date
    market_data : pd.DataFrame
        Market data
    battery_metrics : dict
        Battery performance metrics
    agent_performance : dict, optional
        Agent performance data
    system_results : dict, optional  
        System results
        
    Returns
    -------
    insights : List[ResearchInsight]
        Generated insights
    """
    analyst = AutomatedResearchAnalyst(research_dir)
    
    insights = analyst.analyze_daily_results(
        date=date,
        market_data=market_data,
        battery_metrics=battery_metrics,
        agent_performance=agent_performance or {},
        system_results=system_results or {}
    )
    
    return insights


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import pandas as pd
    from datetime import datetime
    
    # Mock data for testing
    research_dir = Path("research")
    date = datetime.now()
    
    # Create sample market data
    timestamps = pd.date_range(start=date, periods=288, freq='5min')  # Full day
    lmp_data = 30 + 20 * np.sin(np.linspace(0, 2*np.pi, 288)) + np.random.normal(0, 5, 288)
    lmp_data = np.maximum(lmp_data, 0)  # No negative prices
    
    market_data = pd.DataFrame({
        'timestamp': timestamps,
        'lmp_usd': lmp_data
    })
    
    # Mock battery metrics
    battery_metrics = {
        'final_pnl': 150.25,
        'soc_utilization_pct': 65.2,
        'estimated_cycles': 1.2,
        'charge_intervals': 45,
        'discharge_intervals': 52
    }
    
    # Run analysis
    insights = run_automated_analysis(
        research_dir=research_dir,
        date=date,
        market_data=market_data,
        battery_metrics=battery_metrics
    )
    
    print(f"\n🎯 Generated {len(insights)} insights:")
    for insight in insights:
        print(f"  - {insight.title} ({insight.significance} significance)")