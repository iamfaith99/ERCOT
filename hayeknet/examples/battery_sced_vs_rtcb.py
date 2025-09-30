#!/usr/bin/env python3
"""
Compare battery bidding strategies under SCED vs RTC+B market designs.

Research Question:
How would ERCOT battery trading strategies differ under today's market rules 
versus the upcoming RTC+B framework?
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from python.battery_data import BatteryDataClient
from python.battery_model import BatterySimulator, BatterySpecs
from python.market_simulator import compare_market_designs


def main():
    """Run SCED vs RTC+B comparison analysis."""
    print("=" * 80)
    print("🔋 Battery Trading Strategy Comparison")
    print("   SCED (Current) vs RTC+B (December 2025)")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Fetch Historical Data
    # ========================================================================
    print("\n📊 STEP 1: Fetching Historical Market Data")
    print("-" * 80)
    
    # Fetch 7 days of historical data
    start_date = datetime.utcnow() - timedelta(days=7)
    end_date = datetime.utcnow()
    
    client = BatteryDataClient(include_ancillary=True)
    market_data = client.fetch_historical_rtc_data(start_date, end_date)
    
    print(f"✅ Loaded {len(market_data)} intervals")
    print(f"   Period: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
    
    # Compute market statistics
    stats = client.compute_market_statistics(market_data)
    print(f"\n📈 Market Statistics:")
    print(f"   Mean LMP: ${stats['mean_lmp']:.2f}/MWh")
    print(f"   LMP Std: ${stats['std_lmp']:.2f}/MWh")
    print(f"   Max LMP: ${stats['max_lmp']:.2f}/MWh")
    print(f"   Price Spikes (>$100): {stats['spike_frequency']*100:.1f}%")
    if "mean_reg_up" in stats:
        print(f"   Mean Reg Up: ${stats['mean_reg_up']:.2f}/MW")
        print(f"   Mean RRS: ${stats['mean_rrs']:.2f}/MW")
    
    # ========================================================================
    # STEP 2: Define Battery Specifications
    # ========================================================================
    print("\n🔋 STEP 2: Battery Specifications")
    print("-" * 80)
    
    battery_specs = BatterySpecs(
        max_charge_mw=100.0,
        max_discharge_mw=100.0,
        capacity_mwh=200.0,
        min_soc=0.1,
        max_soc=0.9,
        initial_soc=0.5,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        round_trip_efficiency=0.9,
        degradation_cost_per_mwh=5.0,
        can_provide_reg_up=True,
        can_provide_reg_down=True,
        can_provide_rrs=True,
        can_provide_ecrs=True,
    )
    
    print(f"   Power: {battery_specs.max_discharge_mw} MW discharge / {battery_specs.max_charge_mw} MW charge")
    print(f"   Capacity: {battery_specs.capacity_mwh} MWh")
    print(f"   Usable Capacity: {battery_specs.usable_capacity_mwh:.1f} MWh (SOC: {battery_specs.min_soc*100:.0f}% - {battery_specs.max_soc*100:.0f}%)")
    print(f"   Round-trip Efficiency: {battery_specs.round_trip_efficiency*100:.0f}%")
    print(f"   Degradation Cost: ${battery_specs.degradation_cost_per_mwh}/MWh")
    
    # ========================================================================
    # STEP 3: Run Market Simulations
    # ========================================================================
    print("\n🎯 STEP 3: Running Market Simulations")
    print("-" * 80)
    
    # Use subset for faster demo (full week takes ~1 minute)
    sim_data = market_data.iloc[:min(1000, len(market_data))].copy()
    
    print(f"   Simulating {len(sim_data)} intervals ({len(sim_data)/12:.1f} hours)")
    print("   Strategy: Simple Arbitrage (charge low, discharge high, participate in AS)")
    
    comparison_df, summary = compare_market_designs(
        sim_data,
        battery_specs,
        bidding_strategy="simple_arbitrage",
    )
    
    # ========================================================================
    # STEP 4: Analyze Results
    # ========================================================================
    print("\n📊 STEP 4: Analysis Results")
    print("=" * 80)
    
    print(f"\n💰 Revenue Comparison:")
    print(f"   SCED Total Revenue:  ${summary['sced_total_revenue']:,.2f}")
    print(f"   RTC+B Total Revenue: ${summary['rtcb_total_revenue']:,.2f}")
    print(f"   Improvement:         ${summary['revenue_improvement']:,.2f} ({summary['revenue_improvement_pct']:.1f}%)")
    
    print(f"\n⚡ Energy Sources:")
    print(f"   SCED - AS Revenue:  {summary['sced_as_revenue_pct']:.1f}% of total")
    print(f"   RTC+B - AS Revenue: {summary['rtcb_as_revenue_pct']:.1f}% of total")
    
    print(f"\n🔄 Battery Utilization:")
    print(f"   SCED Cycles:  {summary['sced_cycles']:.2f}")
    print(f"   RTC+B Cycles: {summary['rtcb_cycles']:.2f}")
    
    # ========================================================================
    # STEP 5: Visualize Results
    # ========================================================================
    print("\n📈 STEP 5: Generating Visualizations")
    print("-" * 80)
    
    # Create output directory
    output_dir = Path(__file__).resolve().parents[1] / "runs" / "battery_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Revenue Comparison Over Time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cumulative revenue
    sced_data = comparison_df[comparison_df["market_design"] == "SCED"].copy()
    rtcb_data = comparison_df[comparison_df["market_design"] == "RTC+B"].copy()
    
    sced_data["cumulative_revenue"] = sced_data["revenue"].cumsum()
    rtcb_data["cumulative_revenue"] = rtcb_data["revenue"].cumsum()
    
    axes[0, 0].plot(sced_data.index, sced_data["cumulative_revenue"], 
                   label="SCED", color="steelblue", linewidth=2)
    axes[0, 0].plot(rtcb_data.index, rtcb_data["cumulative_revenue"], 
                   label="RTC+B", color="forestgreen", linewidth=2)
    axes[0, 0].set_ylabel("Cumulative Revenue ($)")
    axes[0, 0].set_title("Cumulative Revenue: SCED vs RTC+B")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Revenue distribution
    axes[0, 1].hist([sced_data["revenue"], rtcb_data["revenue"]], 
                   bins=30, label=["SCED", "RTC+B"], alpha=0.7)
    axes[0, 1].set_xlabel("Revenue per Interval ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Revenue Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    
    # SOC comparison
    axes[1, 0].plot(sced_data.index, sced_data["soc"] * 100, 
                   label="SCED", color="steelblue", alpha=0.7)
    axes[1, 0].plot(rtcb_data.index, rtcb_data["soc"] * 100, 
                   label="RTC+B", color="forestgreen", alpha=0.7)
    axes[1, 0].set_ylabel("State of Charge (%)")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_title("Battery State of Charge")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Power output comparison
    axes[1, 1].plot(sced_data.index, sced_data["power_mw"], 
                   label="SCED", color="steelblue", alpha=0.7)
    axes[1, 1].plot(rtcb_data.index, rtcb_data["power_mw"], 
                   label="RTC+B", color="forestgreen", alpha=0.7)
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set_ylabel("Power (MW, +discharge/-charge)")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_title("Battery Power Output")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sced_vs_rtcb_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_dir / 'sced_vs_rtcb_comparison.png'}")
    
    # Plot 2: Revenue Breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    
    breakdown_data = pd.DataFrame({
        "Market Design": ["SCED", "SCED", "RTC+B", "RTC+B"],
        "Revenue Type": ["Energy", "Ancillary Services", "Energy", "Ancillary Services"],
        "Revenue": [
            sced_data["energy_revenue"].sum(),
            sced_data["as_revenue"].sum(),
            rtcb_data["energy_revenue"].sum(),
            rtcb_data["as_revenue"].sum(),
        ]
    })
    
    sns.barplot(data=breakdown_data, x="Market Design", y="Revenue", hue="Revenue Type", ax=ax)
    ax.set_ylabel("Total Revenue ($)")
    ax.set_title("Revenue Breakdown: Energy vs Ancillary Services")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "revenue_breakdown.png", dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_dir / 'revenue_breakdown.png'}")
    
    # Save detailed results
    comparison_df.to_csv(output_dir / "detailed_results.csv", index=False)
    print(f"✅ Saved: {output_dir / 'detailed_results.csv'}")
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    print(f"✅ Saved: {output_dir / 'summary_statistics.csv'}")
    
    # ========================================================================
    # STEP 6: Key Insights
    # ========================================================================
    print("\n🔍 STEP 6: Key Insights")
    print("=" * 80)
    
    improvement_pct = summary['revenue_improvement_pct']
    
    print("\n📌 Main Findings:")
    print(f"   1. RTC+B increases battery revenue by {improvement_pct:.1f}%")
    print(f"   2. Co-optimization allows batteries to provide energy + AS simultaneously")
    print(f"   3. RTC+B: {summary['rtcb_as_revenue_pct']:.1f}% of revenue from AS (vs {summary['sced_as_revenue_pct']:.1f}% in SCED)")
    
    if improvement_pct > 20:
        print(f"   4. ⚠️  Significant economic advantage suggests strong incentive to participate in RTC+B")
    elif improvement_pct > 10:
        print(f"   4. ✅ Moderate improvement justifies strategic adjustment for RTC+B")
    else:
        print(f"   4. ℹ️  Minor improvement; other factors may dominate strategy choice")
    
    print("\n💡 Strategic Implications:")
    print("   • RTC+B enables more efficient battery dispatch")
    print("   • Batteries can capture value from multiple revenue streams")
    print("   • Co-optimization reduces opportunity cost of AS commitments")
    print("   • Price formation expected to be more efficient under RTC+B")
    
    print("\n📅 Next Steps for Research:")
    print("   • Extend analysis to full year of historical data")
    print("   • Test different battery sizes (10MW, 50MW, 200MW, 1000MW)")
    print("   • Incorporate RL-based optimal bidding strategies")
    print("   • Model degradation costs more accurately")
    print("   • Analyze impact of ASDC curves on AS pricing")
    print("   • Study market-wide effects with fleet of batteries")
    
    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print(f"   Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
