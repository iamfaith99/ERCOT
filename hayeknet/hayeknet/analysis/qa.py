"""Research Question Answering System for HayekNet.

This module enables the AI to answer its own research questions by:
1. Analyzing available data (market, battery, system performance)
2. Generating evidence-based answers
3. Quantifying confidence in answers
4. Updating research observations with answers
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ResearchAnswer:
    """Answer to a research question with evidence."""
    question: str
    answer: str
    evidence: Dict[str, Any]
    confidence: float  # 0-1
    data_sources: List[str]
    reasoning: str
    

class ResearchQuestionAnswerer:
    """AI system that answers research questions using available data."""
    
    def __init__(self):
        """Initialize the question answering system."""
        self.answer_cache = {}
        
    def answer_research_questions(
        self,
        questions: List[str],
        market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any],
        system_results: Dict[str, Any],
        date: datetime
    ) -> List[ResearchAnswer]:
        """Answer a list of research questions using available data.
        
        Parameters
        ----------
        questions : List[str]
            Research questions to answer
        market_data : pd.DataFrame
            Market data for analysis
        battery_metrics : dict
            Battery performance metrics
        system_results : dict
            System results and performance data
        date : datetime
            Date of analysis
            
        Returns
        -------
        answers : List[ResearchAnswer]
            Answers to the questions
        """
        answers = []
        
        for question in questions:
            answer = self._answer_single_question(
                question=question,
                market_data=market_data,
                battery_metrics=battery_metrics,
                system_results=system_results,
                date=date
            )
            
            if answer:
                answers.append(answer)
                
        return answers
    
    def _answer_single_question(
        self,
        question: str,
        market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any],
        system_results: Dict[str, Any],
        date: datetime
    ) -> Optional[ResearchAnswer]:
        """Answer a single research question."""
        
        question_lower = question.lower()
        
        # Route to appropriate answering method based on question type
        if "grid condition" in question_lower or "caused.*spike" in question_lower:
            return self._answer_grid_conditions_question(question, market_data, battery_metrics, system_results)
            
        elif "volatility" in question_lower and "caused" in question_lower:
            return self._answer_volatility_cause_question(question, market_data, battery_metrics, system_results)
            
        elif "capitalize" in question_lower or "opportunity" in question_lower:
            return self._answer_opportunity_question(question, market_data, battery_metrics, system_results)
            
        elif "predict" in question_lower or "forecast" in question_lower:
            return self._answer_prediction_question(question, market_data, battery_metrics, system_results)
            
        elif "theoretical maximum" in question_lower or "optimal" in question_lower:
            return self._answer_theoretical_optimal_question(question, market_data, battery_metrics, system_results)
            
        elif "threshold" in question_lower and ("volatility" in question_lower or "profitability" in question_lower):
            return self._answer_threshold_question(question, market_data, battery_metrics, system_results)
            
        elif "soc" in question_lower and ("manage" in question_lower or "strategy" in question_lower):
            return self._answer_soc_management_question(question, market_data, battery_metrics, system_results)
            
        elif "degradation" in question_lower or "cycling" in question_lower:
            return self._answer_degradation_question(question, market_data, battery_metrics, system_results)
            
        else:
            # Generic answer attempt
            return self._answer_generic_question(question, market_data, battery_metrics, system_results)
    
    def _answer_grid_conditions_question(
        self, question: str, market_data: pd.DataFrame, 
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> ResearchAnswer:
        """Answer questions about grid conditions causing price spikes."""
        
        if market_data.empty or 'lmp_usd' not in market_data.columns:
            return None
            
        lmp = market_data['lmp_usd']
        
        # Identify spike periods
        spikes = lmp[lmp > 100]
        negative_prices = lmp[lmp < 0]
        
        # Analyze spike characteristics
        spike_count = len(spikes)
        spike_pct = (spike_count / len(lmp) * 100) if len(lmp) > 0 else 0
        max_spike = spikes.max() if len(spikes) > 0 else 0
        
        # Analyze volatility
        lmp_std = lmp.std()
        lmp_mean = lmp.mean()
        
        # Build evidence-based answer
        causes = []
        
        if spike_pct > 10:
            causes.append("sustained scarcity conditions (>10% of intervals)")
        elif spike_pct > 1:
            causes.append("intermittent scarcity events")
            
        if len(negative_prices) > 0:
            causes.append(f"oversupply periods ({len(negative_prices)} intervals with negative prices)")
            
        if lmp_std > 50:
            causes.append(f"extreme volatility (σ=${lmp_std:.1f})")
        elif lmp_std > 20:
            causes.append(f"high volatility (σ=${lmp_std:.1f})")
            
        # Infer likely grid conditions
        if spike_count > 0 and len(negative_prices) > 0:
            grid_condition = "High renewable penetration with intermittent generation causing rapid swings between oversupply and scarcity"
        elif spike_count > 100:
            grid_condition = "Sustained high demand or reduced generation capacity"
        elif len(negative_prices) > 100:
            grid_condition = "Excess renewable generation overwhelming demand"
        else:
            grid_condition = "Normal market operations with typical supply-demand balancing"
            
        answer_text = f"{grid_condition}. "
        if causes:
            answer_text += f"Contributing factors: {', '.join(causes)}."
            
        evidence = {
            'spike_count': spike_count,
            'spike_percentage': spike_pct,
            'max_spike_price': max_spike,
            'negative_price_count': len(negative_prices),
            'volatility': lmp_std,
            'mean_price': lmp_mean
        }
        
        confidence = 0.7 if spike_count > 10 else 0.5
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            data_sources=['market_data.lmp_usd'],
            reasoning=f"Analyzed {len(lmp)} price intervals. Found {spike_count} spikes (>{100}/MWh) and {len(negative_prices)} negative prices. Volatility σ=${lmp_std:.1f} indicates {'extreme' if lmp_std > 50 else 'moderate'} grid stress."
        )
    
    def _answer_volatility_cause_question(
        self, question: str, market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> ResearchAnswer:
        """Answer questions about causes of market volatility."""
        
        lmp = market_data['lmp_usd']
        lmp_std = lmp.std()
        lmp_cov = (lmp_std / lmp.mean() * 100) if lmp.mean() > 0 else 0
        
        # Analyze price patterns
        price_range = lmp.max() - lmp.min()
        spikes = len(lmp[lmp > 100])
        negative = len(lmp[lmp < 0])
        
        # Determine primary causes
        causes = []
        
        if spikes > 0 and negative > 0:
            causes.append("rapid supply-demand imbalances (both scarcity and oversupply)")
        elif spikes > 0:
            causes.append("scarcity conditions driving prices above $100/MWh")
        elif negative > 0:
            causes.append("oversupply conditions causing negative prices")
            
        if price_range > 500:
            causes.append(f"extreme price swings (${price_range:.0f}/MWh range)")
        elif price_range > 100:
            causes.append(f"significant price variations (${price_range:.0f}/MWh range)")
            
        # Time-of-day analysis
        if 'timestamp' in market_data.columns:
            market_data_copy = market_data.copy()
            market_data_copy['hour'] = pd.to_datetime(market_data_copy['timestamp']).dt.hour
            hourly_std = market_data_copy.groupby('hour')['lmp_usd'].std()
            
            if hourly_std.max() > hourly_std.mean() * 2:
                causes.append("specific hours with extreme volatility (likely peak demand or renewable ramps)")
        
        answer_text = f"High volatility (CoV={lmp_cov:.1f}%, σ=${lmp_std:.1f}) likely caused by: {', '.join(causes)}."
        
        evidence = {
            'coefficient_of_variation': lmp_cov,
            'standard_deviation': lmp_std,
            'price_range': price_range,
            'spike_count': spikes,
            'negative_count': negative
        }
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=0.75,
            data_sources=['market_data.lmp_usd', 'market_data.timestamp'],
            reasoning=f"CoV={lmp_cov:.1f}% indicates {'extreme' if lmp_cov > 50 else 'high'} volatility. Price range of ${price_range:.0f}/MWh with {spikes} spikes and {negative} negative prices suggests rapid supply-demand changes."
        )
    
    def _answer_opportunity_question(
        self, question: str, market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> ResearchAnswer:
        """Answer questions about whether battery capitalized on opportunities."""
        
        pnl = battery_metrics.get('final_pnl', 0)
        soc_util = battery_metrics.get('soc_utilization_pct', 0)
        discharge_intervals = battery_metrics.get('discharge_intervals', 0)
        charge_intervals = battery_metrics.get('charge_intervals', 0)
        
        lmp = market_data['lmp_usd']
        lmp_mean = lmp.mean()
        lmp_std = lmp.std()
        
        # Calculate theoretical opportunities
        high_price_intervals = len(lmp[lmp > (lmp_mean + lmp_std)])
        low_price_intervals = len(lmp[lmp < (lmp_mean - lmp_std)])
        
        # Assess performance
        if pnl > 1000:
            capitalization = "Yes, effectively"
            reason = f"achieved ${pnl:.2f} profit with {soc_util:.1f}% utilization"
        elif pnl > 0:
            capitalization = "Partially"
            reason = f"achieved modest profit (${pnl:.2f}) but only {soc_util:.1f}% utilization suggests missed opportunities"
        elif discharge_intervals == 0:
            capitalization = "No"
            reason = f"never discharged despite {high_price_intervals} high-price intervals"
        else:
            capitalization = "No"
            reason = f"lost ${abs(pnl):.2f} with poor timing or insufficient arbitrage spread"
            
        answer_text = f"{capitalization}. Battery {reason}."
        
        # Calculate missed opportunity estimate
        if discharge_intervals == 0 and high_price_intervals > 0:
            potential_revenue = high_price_intervals * 100 * (lmp_mean + lmp_std) / 12  # $/hour
            answer_text += f" Estimated missed revenue: ${potential_revenue:.0f}."
            
        evidence = {
            'pnl': pnl,
            'soc_utilization': soc_util,
            'discharge_intervals': discharge_intervals,
            'charge_intervals': charge_intervals,
            'high_price_opportunities': high_price_intervals,
            'low_price_opportunities': low_price_intervals
        }
        
        confidence = 0.85
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=confidence,
            data_sources=['battery_metrics', 'market_data.lmp_usd'],
            reasoning=f"Battery discharged {discharge_intervals} times with {high_price_intervals} high-price opportunities available. PnL of ${pnl:.2f} indicates {'good' if pnl > 0 else 'poor'} opportunity capture."
        )
    
    def _answer_prediction_question(
        self, question: str, market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> ResearchAnswer:
        """Answer questions about prediction/forecasting."""
        
        lmp = market_data['lmp_usd']
        
        # Analyze predictability
        if 'timestamp' in market_data.columns:
            market_data_copy = market_data.copy()
            market_data_copy['hour'] = pd.to_datetime(market_data_copy['timestamp']).dt.hour
            hourly_pattern = market_data_copy.groupby('hour')['lmp_usd'].agg(['mean', 'std'])
            
            # Check if there's a clear diurnal pattern
            pattern_strength = (hourly_pattern['mean'].max() - hourly_pattern['mean'].min()) / hourly_pattern['mean'].mean()
            
            if pattern_strength > 0.5:
                predictability = "moderate to high"
                approach = "time-of-day statistical models or simple moving averages"
            elif pattern_strength > 0.2:
                predictability = "moderate"
                approach = "machine learning models (LSTM, gradient boosting) trained on historical patterns"
            else:
                predictability = "low"
                approach = "probabilistic forecasting with wide uncertainty bounds"
        else:
            predictability = "unknown"
            approach = "insufficient temporal data for assessment"
            
        # Check volatility for predictability
        lmp_cov = (lmp.std() / lmp.mean() * 100) if lmp.mean() > 0 else 0
        
        if lmp_cov > 50:
            answer_text = f"Prediction is challenging due to extreme volatility (CoV={lmp_cov:.1f}%). Recommend {approach} with ensemble methods and short-term (5-15 min) horizons."
        elif lmp_cov > 20:
            answer_text = f"Prediction has {predictability} confidence. Use {approach} with rolling re-training on recent data."
        else:
            answer_text = f"Prediction is relatively straightforward given low volatility. Use {approach}."
            
        evidence = {
            'coefficient_of_variation': lmp_cov,
            'predictability_assessment': predictability,
            'recommended_approach': approach
        }
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=0.6,
            data_sources=['market_data'],
            reasoning=f"Volatility CoV={lmp_cov:.1f}% suggests {'high' if lmp_cov > 50 else 'moderate'} prediction difficulty. Pattern analysis indicates {predictability} predictability."
        )
    
    def _answer_theoretical_optimal_question(
        self, question: str, market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> ResearchAnswer:
        """Answer questions about theoretical maximum or optimal performance."""
        
        lmp = market_data['lmp_usd']
        
        # Calculate theoretical maximum arbitrage profit
        # Assume perfect foresight: charge at lowest prices, discharge at highest
        lmp_sorted = lmp.sort_values()
        
        # Assume 100MW battery, 400MWh capacity, 90% efficiency
        capacity_mwh = 400
        power_mw = 100
        efficiency = 0.90
        
        # Estimate number of full cycles possible
        total_intervals = len(lmp)
        intervals_per_cycle = (capacity_mwh / power_mw) * 2  # Charge + discharge
        max_cycles = total_intervals / intervals_per_cycle
        
        # Calculate optimal arbitrage (charge at bottom 30%, discharge at top 30%)
        charge_cutoff = int(len(lmp_sorted) * 0.3)
        discharge_cutoff = int(len(lmp_sorted) * 0.7)
        
        avg_charge_price = lmp_sorted[:charge_cutoff].mean()
        avg_discharge_price = lmp_sorted[discharge_cutoff:].mean()
        
        theoretical_profit = (avg_discharge_price - avg_charge_price) * capacity_mwh * efficiency * min(max_cycles, 2)
        
        actual_pnl = battery_metrics.get('final_pnl', 0)
        actual_cycles = battery_metrics.get('estimated_cycles', 0)
        
        efficiency_pct = (actual_pnl / theoretical_profit * 100) if theoretical_profit != 0 else 0
        
        answer_text = f"Theoretical maximum profit: ${theoretical_profit:.2f} (assuming perfect foresight, {min(max_cycles, 2):.2f} cycles). "
        answer_text += f"Actual performance: ${actual_pnl:.2f} ({efficiency_pct:.1f}% of theoretical max). "
        
        if efficiency_pct < 0:
            answer_text += "Negative returns indicate poor timing or adverse market conditions."
        elif efficiency_pct < 20:
            answer_text += "Significant improvement potential through better price forecasting and timing."
        elif efficiency_pct < 50:
            answer_text += "Moderate performance with room for optimization."
        else:
            answer_text += "Strong performance approaching theoretical limits."
            
        evidence = {
            'theoretical_max_profit': theoretical_profit,
            'actual_profit': actual_pnl,
            'efficiency_percentage': efficiency_pct,
            'theoretical_cycles': min(max_cycles, 2),
            'actual_cycles': actual_cycles,
            'optimal_charge_price': avg_charge_price,
            'optimal_discharge_price': avg_discharge_price
        }
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=0.8,
            data_sources=['market_data.lmp_usd', 'battery_metrics'],
            reasoning=f"Calculated theoretical optimum assuming perfect foresight: charge at avg ${avg_charge_price:.2f}/MWh, discharge at ${avg_discharge_price:.2f}/MWh. Actual achieved {efficiency_pct:.1f}% of theoretical."
        )
    
    def _answer_threshold_question(
        self, question: str, market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> ResearchAnswer:
        """Answer questions about volatility/profitability thresholds."""
        
        lmp = market_data['lmp_usd']
        lmp_std = lmp.std()
        lmp_cov = (lmp_std / lmp.mean() * 100) if lmp.mean() > 0 else 0
        pnl = battery_metrics.get('final_pnl', 0)
        
        # Estimate profitability threshold
        # For arbitrage to be profitable, need spread > degradation + transaction costs
        min_spread_for_profit = 15.0  # $/MWh (includes efficiency losses + degradation)
        
        actual_spread = lmp.max() - lmp.min()
        
        answer_text = f"Based on today's data (CoV={lmp_cov:.1f}%, PnL=${pnl:.2f}), "
        
        if pnl > 0:
            answer_text += f"profitability threshold appears to be around CoV≈{lmp_cov * 0.8:.1f}% or spread>${actual_spread * 0.8:.0f}/MWh. "
        else:
            answer_text += f"volatility (CoV={lmp_cov:.1f}%) was insufficient for profitability. Recommend minimum CoV>20% and spread>${min_spread_for_profit:.0f}/MWh. "
            
        answer_text += f"For 90% round-trip efficiency battery, need spread >${min_spread_for_profit:.0f}/MWh to cover costs and degradation."
        
        evidence = {
            'observed_cov': lmp_cov,
            'observed_spread': actual_spread,
            'profitability': pnl > 0,
            'recommended_min_spread': min_spread_for_profit,
            'recommended_min_cov': 20.0
        }
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=0.65,
            data_sources=['market_data', 'battery_metrics'],
            reasoning=f"Actual: CoV={lmp_cov:.1f}%, spread=${actual_spread:.0f}/MWh, PnL=${pnl:.2f}. For 90% efficiency, min spread≈$15/MWh needed."
        )
    
    def _answer_soc_management_question(
        self, question: str, market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> ResearchAnswer:
        """Answer questions about SOC management strategy."""
        
        soc_util = battery_metrics.get('soc_utilization_pct', 0)
        soc_mean = battery_metrics.get('soc_mean', 0.5)
        cycles = battery_metrics.get('estimated_cycles', 0)
        pnl = battery_metrics.get('final_pnl', 0)
        
        lmp_cov = (market_data['lmp_usd'].std() / market_data['lmp_usd'].mean() * 100) if market_data['lmp_usd'].mean() > 0 else 0
        
        # Optimal SOC strategy depends on market type
        if lmp_cov > 50:  # High volatility
            optimal_strategy = "Maintain SOC around 50% to enable bidirectional arbitrage. Target 60-80% utilization with 1-2 cycles/day."
        elif lmp_cov > 20:  # Moderate volatility
            optimal_strategy = "Maintain SOC 40-60% with opportunistic full cycles when spreads >$20/MWh. Target 40-60% utilization."
        else:  # Low volatility
            optimal_strategy = "Conservative SOC management (30-70%) due to limited arbitrage opportunities. Prioritize ancillary services."
            
        assessment = f"Today's utilization ({soc_util:.1f}%) was "
        if soc_util < 30:
            assessment += "too conservative - battery underutilized."
        elif soc_util > 80:
            assessment += "aggressive - high cycling may increase degradation."
        else:
            assessment += "reasonable for the volatility level."
            
        answer_text = f"{optimal_strategy} {assessment}"
        
        evidence = {
            'soc_utilization': soc_util,
            'cycles': cycles,
            'market_volatility': lmp_cov,
            'profitability': pnl
        }
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=0.75,
            data_sources=['battery_metrics', 'market_data'],
            reasoning=f"For CoV={lmp_cov:.1f}% market, optimal utilization is 40-80%. Actual: {soc_util:.1f}% with {cycles:.2f} cycles."
        )
    
    def _answer_degradation_question(
        self, question: str, market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> ResearchAnswer:
        """Answer questions about degradation and cycling costs."""
        
        cycles = battery_metrics.get('estimated_cycles', 0)
        pnl = battery_metrics.get('final_pnl', 0)
        
        # Typical degradation cost: $5/MWh throughput
        degradation_cost_per_mwh = 5.0
        capacity_mwh = 400
        
        degradation_cost = cycles * capacity_mwh * degradation_cost_per_mwh
        net_pnl_after_degradation = pnl - degradation_cost
        
        answer_text = f"With {cycles:.2f} cycles today, degradation cost ≈ ${degradation_cost:.2f} (at ${degradation_cost_per_mwh}/MWh throughput). "
        answer_text += f"Net PnL after degradation: ${net_pnl_after_degradation:.2f}. "
        
        if cycles < 1:
            answer_text += "Low cycling minimizes degradation but may miss revenue opportunities."
        elif cycles < 2:
            answer_text += "Cycling rate is sustainable for daily arbitrage operations."
        else:
            answer_text += "High cycling may accelerate degradation - ensure revenue justifies wear."
            
        evidence = {
            'cycles': cycles,
            'degradation_cost': degradation_cost,
            'gross_pnl': pnl,
            'net_pnl_after_degradation': net_pnl_after_degradation
        }
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence=evidence,
            confidence=0.8,
            data_sources=['battery_metrics'],
            reasoning=f"{cycles:.2f} cycles × {capacity_mwh} MWh × ${degradation_cost_per_mwh}/MWh = ${degradation_cost:.2f} degradation cost."
        )
    
    def _answer_generic_question(
        self, question: str, market_data: pd.DataFrame,
        battery_metrics: Dict[str, Any], system_results: Dict[str, Any]
    ) -> Optional[ResearchAnswer]:
        """Attempt to answer questions that don't match specific patterns."""
        
        # Provide generic insight based on available data
        pnl = battery_metrics.get('final_pnl', 0)
        lmp_std = market_data['lmp_usd'].std() if not market_data.empty else 0
        
        answer_text = f"Based on today's data: Market volatility σ=${lmp_std:.2f}/MWh, Battery PnL=${pnl:.2f}. "
        answer_text += "This question requires more specific analysis or additional data for a complete answer."
        
        return ResearchAnswer(
            question=question,
            answer=answer_text,
            evidence={'lmp_volatility': lmp_std, 'battery_pnl': pnl},
            confidence=0.3,
            data_sources=['market_data', 'battery_metrics'],
            reasoning="Generic answer - question pattern not recognized for detailed analysis."
        )
    
    def save_answers(self, answers: List[ResearchAnswer], output_file: Path):
        """Save answers to JSON file."""
        answers_data = []
        for answer in answers:
            answers_data.append({
                'question': answer.question,
                'answer': answer.answer,
                'evidence': answer.evidence,
                'confidence': answer.confidence,
                'data_sources': answer.data_sources,
                'reasoning': answer.reasoning
            })
            
        with open(output_file, 'w') as f:
            json.dump(answers_data, f, indent=2)
            
        print(f"💾 Saved {len(answers)} answers to {output_file}")
    
    def generate_qa_summary(self, answers: List[ResearchAnswer], date: datetime) -> str:
        """Generate markdown summary of Q&A session."""
        summary = f"# Research Questions & Answers - {date.strftime('%Y-%m-%d')}\n\n"
        summary += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        summary += "---\n\n"
        
        high_conf_answers = [a for a in answers if a.confidence >= 0.7]
        medium_conf_answers = [a for a in answers if 0.4 <= a.confidence < 0.7]
        low_conf_answers = [a for a in answers if a.confidence < 0.4]
        
        if high_conf_answers:
            summary += "## ✅ High Confidence Answers\n\n"
            for answer in high_conf_answers:
                summary += f"### Q: {answer.question}\n\n"
                summary += f"**A**: {answer.answer}\n\n"
                summary += f"**Confidence**: {answer.confidence*100:.0f}%\n\n"
                summary += f"**Evidence**: {', '.join(f'{k}={v}' for k, v in list(answer.evidence.items())[:3])}\n\n"
                summary += "---\n\n"
                
        if medium_conf_answers:
            summary += "## ⚠️ Medium Confidence Answers\n\n"
            for answer in medium_conf_answers:
                summary += f"### Q: {answer.question}\n\n"
                summary += f"**A**: {answer.answer}\n\n"
                summary += f"**Confidence**: {answer.confidence*100:.0f}%\n\n"
                summary += "---\n\n"
                
        if low_conf_answers:
            summary += "## ❓ Low Confidence Answers (Needs More Data)\n\n"
            for answer in low_conf_answers:
                summary += f"### Q: {answer.question}\n\n"
                summary += f"**A**: {answer.answer}\n\n"
                summary += f"**Confidence**: {answer.confidence*100:.0f}%\n\n"
                summary += "---\n\n"
                
        summary += f"\n**Total Questions Answered**: {len(answers)}\n"
        summary += f"**Average Confidence**: {np.mean([a.confidence for a in answers])*100:.0f}%\n"
        
        return summary


def answer_and_save_questions(
    questions: List[str],
    market_data: pd.DataFrame,
    battery_metrics: Dict[str, Any],
    system_results: Dict[str, Any],
    research_dir: Path,
    date: datetime
) -> List[ResearchAnswer]:
    """Answer research questions and save results.
    
    Convenience function for the full Q&A pipeline.
    """
    answerer = ResearchQuestionAnswerer()
    
    answers = answerer.answer_research_questions(
        questions=questions,
        market_data=market_data,
        battery_metrics=battery_metrics,
        system_results=system_results,
        date=date
    )
    
    # Save answers
    qa_dir = research_dir / 'qa'
    qa_dir.mkdir(parents=True, exist_ok=True)
    
    answers_file = qa_dir / f"answers_{date.strftime('%Y-%m-%d')}.json"
    answerer.save_answers(answers, answers_file)
    
    # Generate summary
    summary = answerer.generate_qa_summary(answers, date)
    summary_file = qa_dir / f"qa_summary_{date.strftime('%Y-%m-%d')}.md"
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"📝 Saved Q&A summary to {summary_file}")
    
    return answers


if __name__ == "__main__":
    # Example usage
    from datetime import datetime
    import pandas as pd
    import numpy as np
    
    # Mock data
    market_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-10-02', periods=288, freq='5min'),
        'lmp_usd': 30 + 50 * np.sin(np.linspace(0, 2*np.pi, 288)) + np.random.normal(0, 10, 288)
    })
    
    battery_metrics = {
        'final_pnl': -150.0,
        'soc_utilization_pct': 35.0,
        'estimated_cycles': 0.4,
        'discharge_intervals': 0,
        'charge_intervals': 20
    }
    
    questions = [
        "What grid conditions caused this spike?",
        "Did our battery capitalize on this opportunity?",
        "What's the theoretical maximum profit?"
    ]
    
    answers = answer_and_save_questions(
        questions=questions,
        market_data=market_data,
        battery_metrics=battery_metrics,
        system_results={},
        research_dir=Path("research"),
        date=datetime.now()
    )
    
    print(f"\n✅ Answered {len(answers)} questions")