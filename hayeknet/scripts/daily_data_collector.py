#!/usr/bin/env python3
"""Daily automated ERCOT data collection and analysis.

This script runs daily to:
1. Fetch new ERCOT data from the last 24 hours
2. Append to the master historical archive
3. Generate daily analysis reports
4. Clean up old cache files

Designed to be run via cron or scheduled task.
"""
from __future__ import annotations

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.data import ERCOTDataClient


def setup_directories():
    """Ensure all required directories exist."""
    base_dir = Path(__file__).resolve().parents[1]
    
    dirs = {
        'archive': base_dir / 'data' / 'archive' / 'ercot_lmp',
        'reports': base_dir / 'data' / 'reports',
        'logs': base_dir / 'data' / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def fetch_daily_data(client: ERCOTDataClient, max_reports: int = 300) -> pd.DataFrame:
    """Fetch last 24 hours of data.
    
    Args:
        client: ERCOT data client
        max_reports: Max reports to fetch (~288 for 24 hours of 5-min data)
    
    Returns:
        DataFrame with new data
    """
    print(f"\n{'='*80}")
    print(f"Daily Data Collection - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    print(f"📥 Fetching last 24 hours of ERCOT data...")
    print(f"   Target: ~{max_reports} reports (5-min intervals)")
    
    df = client.fetch_historical_data(
        report_type="real_time_lmp",
        max_reports=max_reports,
        use_cache=True
    )
    
    if df.empty:
        print("⚠️  No new data retrieved")
        return df
    
    # Parse timestamp
    if 'SCEDTimestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['SCEDTimestamp'])
    
    print(f"\n✅ Retrieved {len(df):,} observations")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def append_to_archive(df: pd.DataFrame, archive_dir: Path) -> dict:
    """Append new data to master archive.
    
    Args:
        df: New data to append
        archive_dir: Directory containing archive files
    
    Returns:
        Dictionary with archive statistics
    """
    if df.empty:
        return {'status': 'no_data'}
    
    print(f"\n📦 Updating master archive...")
    
    # Determine archive file (one per month)
    latest_date = df['timestamp'].max()
    
    # Try parquet first, fall back to pickle if pyarrow not available
    try:
        archive_file = archive_dir / f"ercot_lmp_{latest_date.year}_{latest_date.month:02d}.parquet"
        
        # Load existing archive if it exists
        if archive_file.exists():
            print(f"   Loading existing archive: {archive_file.name}")
            existing_df = pd.read_parquet(archive_file)
            print(f"   Existing records: {len(existing_df):,}")
            
            # Combine and deduplicate
            combined = pd.concat([existing_df, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp', 'settlement_point'])
            combined = combined.sort_values('timestamp').reset_index(drop=True)
            
            new_records = len(combined) - len(existing_df)
            print(f"   Added {new_records:,} new unique records")
        else:
            print(f"   Creating new archive: {archive_file.name}")
            combined = df.sort_values('timestamp').reset_index(drop=True)
            new_records = len(combined)
        
        # Save updated archive
        combined.to_parquet(archive_file, compression='snappy', index=False)
        
    except ImportError:
        # Fallback to pickle if parquet not available
        print("   ⚠️  pyarrow not available, using pickle format")
        archive_file = archive_dir / f"ercot_lmp_{latest_date.year}_{latest_date.month:02d}.pkl"
        
        if archive_file.exists():
            print(f"   Loading existing archive: {archive_file.name}")
            existing_df = pd.read_pickle(archive_file)
            print(f"   Existing records: {len(existing_df):,}")
            
            combined = pd.concat([existing_df, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp', 'settlement_point'])
            combined = combined.sort_values('timestamp').reset_index(drop=True)
            
            new_records = len(combined) - len(existing_df)
            print(f"   Added {new_records:,} new unique records")
        else:
            print(f"   Creating new archive: {archive_file.name}")
            combined = df.sort_values('timestamp').reset_index(drop=True)
            new_records = len(combined)
        
        # Save updated archive
        combined.to_pickle(archive_file)
    
    stats = {
        'status': 'success',
        'archive_file': str(archive_file.name),
        'total_records': len(combined),
        'new_records': new_records,
        'date_range': {
            'start': str(combined['timestamp'].min()),
            'end': str(combined['timestamp'].max())
        }
    }
    
    print(f"   Total records in archive: {len(combined):,}")
    print(f"   Archive file: {archive_file.name}")
    
    return stats


def generate_daily_report(df: pd.DataFrame, reports_dir: Path) -> dict:
    """Generate daily analysis report.
    
    Args:
        df: Data to analyze
        reports_dir: Directory to save reports
    
    Returns:
        Dictionary with report statistics
    """
    if df.empty:
        return {'status': 'no_data'}
    
    print(f"\n📊 Generating daily analysis report...")
    
    today = datetime.now().date()
    report_file = reports_dir / f"daily_report_{today.isoformat()}.md"
    
    # Extract hub data
    hubs = df[df['settlement_point'].str.startswith('HB_')].copy()
    
    if hubs.empty:
        print("⚠️  No hub data available for analysis")
        return {'status': 'no_hub_data'}
    
    # Compute system average
    system_lmp = hubs.groupby('timestamp')['lmp_usd'].mean().reset_index()
    system_lmp.columns = ['timestamp', 'lmp']
    
    # Statistics
    mean_lmp = system_lmp['lmp'].mean()
    std_lmp = system_lmp['lmp'].std()
    min_lmp = system_lmp['lmp'].min()
    max_lmp = system_lmp['lmp'].max()
    q95_lmp = system_lmp['lmp'].quantile(0.95)
    
    # Hub breakdown
    hub_stats = hubs.groupby('settlement_point')['lmp_usd'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)
    
    # High-price events
    high_price = system_lmp[system_lmp['lmp'] > 100]
    high_price_pct = 100 * len(high_price) / len(system_lmp) if len(system_lmp) > 0 else 0
    
    # Volatility
    system_lmp['lmp_change'] = system_lmp['lmp'].diff()
    avg_abs_change = abs(system_lmp['lmp_change']).mean()
    max_increase = system_lmp['lmp_change'].max()
    max_decrease = system_lmp['lmp_change'].min()
    
    # Generate markdown report
    report_content = f"""# ERCOT Daily Analysis Report
## {today.strftime('%B %d, %Y')}

---

## 📊 System-Wide Statistics

| Metric | Value |
|--------|-------|
| **Mean LMP** | ${mean_lmp:.2f}/MWh |
| **Median LMP** | ${system_lmp['lmp'].median():.2f}/MWh |
| **Std Dev** | ${std_lmp:.2f}/MWh |
| **Min LMP** | ${min_lmp:.2f}/MWh |
| **Max LMP** | ${max_lmp:.2f}/MWh |
| **95th Percentile** | ${q95_lmp:.2f}/MWh |

### Time Coverage
- **Start**: {system_lmp['timestamp'].min()}
- **End**: {system_lmp['timestamp'].max()}
- **Intervals**: {len(system_lmp):,} (5-minute periods)
- **Duration**: {(system_lmp['timestamp'].max() - system_lmp['timestamp'].min()).total_seconds() / 3600:.1f} hours

---

## 📈 Price Volatility

| Metric | Value |
|--------|-------|
| **Mean Absolute Change** | ${avg_abs_change:.2f}/MWh |
| **Max Single Increase** | ${max_increase:.2f}/MWh |
| **Max Single Decrease** | ${max_decrease:.2f}/MWh |
| **Coefficient of Variation** | {(std_lmp/mean_lmp)*100:.2f}% |

---

## 🏢 Hub-by-Hub Comparison

### Top 5 Most Expensive Hubs

| Rank | Hub | Avg LMP | Std Dev | Min | Max |
|------|-----|---------|---------|-----|-----|
"""
    
    top_hubs = hub_stats.nlargest(5, 'mean')
    for i, (hub, row) in enumerate(top_hubs.iterrows(), 1):
        report_content += f"| {i} | {hub} | ${row['mean']:.2f} | ${row['std']:.2f} | ${row['min']:.2f} | ${row['max']:.2f} |\n"
    
    report_content += f"""
### All Hubs Summary

| Hub | Observations | Avg LMP | Std Dev |
|-----|--------------|---------|---------|
"""
    
    for hub, row in hub_stats.iterrows():
        report_content += f"| {hub} | {row['count']:,} | ${row['mean']:.2f} | ${row['std']:.2f} |\n"
    
    report_content += f"""
---

## ⚠️ High-Price Events (>$100/MWh)

- **Occurrences**: {len(high_price):,}
- **% of Time**: {high_price_pct:.2f}%
"""
    
    if not high_price.empty:
        peak_time = high_price.loc[high_price['lmp'].idxmax(), 'timestamp']
        peak_price = high_price['lmp'].max()
        report_content += f"- **Peak Price**: ${peak_price:.2f}/MWh at {peak_time}\n"
    else:
        report_content += "- **Status**: No high-price events detected\n"
    
    report_content += f"""
---

## 📉 Price Distribution

"""
    
    # Price bins
    bins = [0, 25, 50, 75, 100, 150, 200, float('inf')]
    labels = ['$0-25', '$25-50', '$50-75', '$75-100', '$100-150', '$150-200', '$200+']
    system_lmp['price_bin'] = pd.cut(system_lmp['lmp'], bins=bins, labels=labels)
    dist = system_lmp['price_bin'].value_counts().sort_index()
    
    report_content += "| Price Range | Count | Percentage |\n"
    report_content += "|-------------|-------|------------|\n"
    
    for price_range, count in dist.items():
        pct = 100 * count / len(system_lmp)
        report_content += f"| {price_range} | {count:,} | {pct:.1f}% |\n"
    
    report_content += f"""
---

## 📋 Data Quality

| Metric | Value |
|--------|-------|
| **Total Observations** | {len(df):,} |
| **Unique Settlement Points** | {df['settlement_point'].nunique():,} |
| **Hub Observations** | {len(hubs):,} |
| **Missing Values** | {df.isnull().sum().sum():,} |
| **Data Completeness** | {100 * (1 - df.isnull().sum().sum() / df.size):.2f}% |

---

## 🔄 Collection Info

- **Collection Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Report Generated**: Automated daily collection
- **Next Collection**: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')} (automated)

---

*Report auto-generated by HayekNet daily data collector*
"""
    
    # Save report
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"   Report saved: {report_file.name}")
    
    # Also save CSV summary
    csv_file = reports_dir / f"daily_summary_{today.isoformat()}.csv"
    system_lmp.to_csv(csv_file, index=False)
    print(f"   CSV saved: {csv_file.name}")
    
    return {
        'status': 'success',
        'report_file': str(report_file.name),
        'csv_file': str(csv_file.name),
        'mean_lmp': round(mean_lmp, 2),
        'std_lmp': round(std_lmp, 2),
        'high_price_events': len(high_price)
    }


def cleanup_cache(client: ERCOTDataClient, days: int = 7):
    """Clean up old cache files.
    
    Args:
        client: ERCOT data client
        days: Keep files newer than this many days
    """
    print(f"\n🗑️  Cleaning cache (keeping files < {days} days old)...")
    count = client.clear_cache(older_than_days=days)
    print(f"   Removed {count} old cache files")


def save_log(dirs: dict, stats: dict):
    """Save execution log.
    
    Args:
        dirs: Directory paths
        stats: Statistics from execution
    """
    log_file = dirs['logs'] / f"collection_log_{datetime.now().date().isoformat()}.json"
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'status': 'success',
        'archive_stats': stats.get('archive', {}),
        'report_stats': stats.get('report', {})
    }
    
    # Append to daily log
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = json.load(f)
        logs.append(log_entry)
    else:
        logs = [log_entry]
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"\n📝 Execution log saved: {log_file.name}")


def main():
    """Main execution."""
    try:
        # Setup
        dirs = setup_directories()
        client = ERCOTDataClient()
        
        # Fetch data
        df = fetch_daily_data(client, max_reports=300)
        
        if df.empty:
            print("\n⚠️  No data to process. Exiting.")
            return
        
        # Archive
        archive_stats = append_to_archive(df, dirs['archive'])
        
        # Generate report
        report_stats = generate_daily_report(df, dirs['reports'])
        
        # Cleanup
        cleanup_cache(client, days=7)
        
        # Log
        save_log(dirs, {'archive': archive_stats, 'report': report_stats})
        
        print(f"\n{'='*80}")
        print("✅ Daily collection complete!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error during daily collection: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
