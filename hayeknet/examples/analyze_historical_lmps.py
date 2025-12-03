#!/usr/bin/env python3
"""Example: Analyze historical ERCOT LMP data.

This script demonstrates how to:
1. Fetch historical LMP data from ERCOT
2. Compute hub-averaged prices
3. Analyze price statistics and patterns
4. Create visualizations (optional)
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hayeknet.data.client import ERCOTDataClient


def main():
    """Analyze historical LMP data."""
    print("\n" + "="*80)
    print("ERCOT Historical LMP Analysis")
    print("="*80 + "\n")
    
    # Initialize client
    print("🚀 Initializing ERCOT Data Client...")
    client = ERCOTDataClient()
    
    # Fetch recent historical data
    print("\n📥 Fetching historical LMP data (last 48 hours)...")
    df = client.fetch_historical_data(
        report_type="real_time_lmp",
        start_date=datetime.now() - timedelta(hours=48),
        max_reports=50,  # Limit to 50 most recent reports
        use_cache=True
    )
    
    if df.empty:
        print("❌ No data retrieved. Exiting.")
        return
    
    print(f"\n✅ Retrieved {len(df):,} price observations")
    
    # Parse timestamp
    if 'SCEDTimestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['SCEDTimestamp'])
    
    # Identify hub data (major trading hubs)
    hub_prefixes = ['HB_HOUSTON', 'HB_NORTH', 'HB_SOUTH', 'HB_WEST', 'HB_BUSAVG', 'HB_PAN']
    df['is_hub'] = df['settlement_point'].apply(
        lambda x: any(x.startswith(prefix) for prefix in hub_prefixes)
    )
    
    hub_df = df[df['is_hub']].copy()
    
    print(f"\n📍 Trading Hubs Found:")
    print(f"   - Total hub observations: {len(hub_df):,}")
    print(f"   - Unique hubs: {hub_df['settlement_point'].nunique()}")
    print(f"   - Hub names: {', '.join(sorted(hub_df['settlement_point'].unique()[:10]))}")
    
    # Compute system-wide average LMP
    system_lmp = hub_df.groupby('timestamp')['lmp_usd'].mean().reset_index()
    system_lmp.columns = ['timestamp', 'system_lmp']
    
    print(f"\n📊 System-Wide LMP Statistics:")
    print(f"   - Time Range: {system_lmp['timestamp'].min()} to {system_lmp['timestamp'].max()}")
    print(f"   - Number of 5-min intervals: {len(system_lmp)}")
    print(f"   - Mean LMP: ${system_lmp['system_lmp'].mean():.2f}/MWh")
    print(f"   - Median LMP: ${system_lmp['system_lmp'].median():.2f}/MWh")
    print(f"   - Std Dev: ${system_lmp['system_lmp'].std():.2f}/MWh")
    print(f"   - Min LMP: ${system_lmp['system_lmp'].min():.2f}/MWh")
    print(f"   - Max LMP: ${system_lmp['system_lmp'].max():.2f}/MWh")
    print(f"   - 95th percentile: ${system_lmp['system_lmp'].quantile(0.95):.2f}/MWh")
    
    # Analyze volatility
    system_lmp['lmp_change'] = system_lmp['system_lmp'].diff()
    system_lmp['lmp_pct_change'] = system_lmp['system_lmp'].pct_change() * 100
    
    print(f"\n📈 Price Volatility:")
    print(f"   - Mean absolute change: ${abs(system_lmp['lmp_change']).mean():.2f}/MWh")
    print(f"   - Max increase: ${system_lmp['lmp_change'].max():.2f}/MWh")
    print(f"   - Max decrease: ${system_lmp['lmp_change'].min():.2f}/MWh")
    print(f"   - Std of % change: {system_lmp['lmp_pct_change'].std():.2f}%")
    
    # Hub comparison
    print(f"\n🏢 Hub-by-Hub Analysis:")
    hub_stats = hub_df.groupby('settlement_point')['lmp_usd'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)
    
    # Show top 5 hubs by average price
    top_hubs = hub_stats.nlargest(5, 'mean')
    print("\n   Top 5 Most Expensive Hubs (by average LMP):")
    for idx, (hub, row) in enumerate(top_hubs.iterrows(), 1):
        print(f"   {idx}. {hub:20s} | Avg: ${row['mean']:6.2f}/MWh | Std: ${row['std']:5.2f} | Range: ${row['min']:6.2f}-${row['max']:6.2f}")
    
    # Price distribution
    print(f"\n📉 Price Distribution:")
    bins = [0, 25, 50, 75, 100, 150, 200, float('inf')]
    labels = ['$0-25', '$25-50', '$50-75', '$75-100', '$100-150', '$150-200', '$200+']
    system_lmp['price_bin'] = pd.cut(system_lmp['system_lmp'], bins=bins, labels=labels)
    dist = system_lmp['price_bin'].value_counts().sort_index()
    
    for price_range, count in dist.items():
        pct = 100 * count / len(system_lmp)
        bar = '█' * int(pct / 2)
        print(f"   {price_range:10s} | {count:4d} ({pct:5.1f}%) {bar}")
    
    # Identify high-price events (>$100/MWh)
    high_price = system_lmp[system_lmp['system_lmp'] > 100]
    if not high_price.empty:
        print(f"\n⚠️  High Price Events (>$100/MWh):")
        print(f"   - Occurrences: {len(high_price)}")
        print(f"   - % of time: {100 * len(high_price) / len(system_lmp):.2f}%")
        print(f"   - Peak price: ${high_price['system_lmp'].max():.2f}/MWh at {high_price.loc[high_price['system_lmp'].idxmax(), 'timestamp']}")
    
    # Export summary
    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "historical_lmp_summary.csv"
    system_lmp.to_csv(output_file, index=False)
    print(f"\n💾 Exported time series to: {output_file}")
    
    print("\n" + "="*80)
    print("✅ Analysis Complete")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
