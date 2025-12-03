#!/usr/bin/env python3
"""Enhanced Daily ERCOT Data Collection with Multi-Source Strategy.

This script implements comprehensive data collection using multiple ERCOT sources:
1. Real-Time LMP (5-min, 5.1 days retention) - PRIMARY
2. Day-Ahead Market LMP (hourly, 7.1 days retention) - FALLBACK
3. System Lambda (5-min, 7.1 days retention) - SUPPLEMENT  
4. Real-Time Load (hourly, 7.1 days retention) - CONTEXT
5. Ancillary Prices (12+ years retention) - RISK MODELING

Designed to be run via cron or scheduled task with intelligent fallbacks.
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hayeknet.data.client import ERCOTDataClient


def setup_directories():
    """Ensure all required directories exist."""
    base_dir = Path(__file__).resolve().parents[1]
    
    dirs = {
        'archive': base_dir / 'data' / 'archive' / 'ercot_lmp',
        'multi_source': base_dir / 'data' / 'archive' / 'multi_source',
        'ancillary': base_dir / 'data' / 'archive' / 'ancillary',
        'reports': base_dir / 'data' / 'reports',
        'logs': base_dir / 'data' / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def fetch_daily_data_multi_source(client: ERCOTDataClient, max_reports: int = 300) -> dict:
    """Fetch data from multiple ERCOT sources with intelligent fallbacks.
    
    Args:
        client: ERCOT data client
        max_reports: Max reports to fetch per source
    
    Returns:
        Dictionary with data from each source
    """
    print(f"\n{'='*80}")
    print(f"Enhanced Multi-Source Data Collection - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Configure data sources with priorities
    data_sources = {
        'real_time_lmp': {
            'report_id': '12300',
            'priority': 1,  # Primary source
            'description': 'Real-Time LMP (5-min intervals, 5.1d retention)'
        },
        'dam_lmp': {
            'report_id': '12301', 
            'priority': 2,  # Fallback for missing RT data
            'description': 'Day-Ahead Market LMP (hourly, 7.1d retention)'
        },
        'system_lambda': {
            'report_id': '12301',
            'priority': 3,  # Supplemental data
            'description': 'System Lambda (marginal cost, 7.1d retention)'
        },
        'real_time_load': {
            'report_id': '13071',
            'priority': 4,  # Context data
            'description': 'Real-Time Load (hourly, 7.1d retention)'
        },
        'ancillary_prices': {
            'report_id': '13060',
            'priority': 5,  # Risk modeling data
            'description': 'Ancillary Service Prices (12+ years retention)'
        }
    }
    
    collection_results = {}
    
    # Try each source in priority order
    for source_name, config in sorted(data_sources.items(), key=lambda x: x[1]['priority']):
        print(f"📡 Fetching from {config['description']}...")
        
        # Configure source
        client.endpoints.report_type_ids[source_name] = config['report_id']
        
        try:
            df = client.fetch_historical_data(
                report_type=source_name,
                max_reports=max_reports if source_name != 'ancillary_prices' else 20,  # Fewer for ancillary
                use_cache=True
            )
            
            if not df.empty:
                # Standardize timestamp column
                if 'timestamp' not in df.columns:
                    if 'SCEDTimestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['SCEDTimestamp'])
                    elif 'DeliveryDate' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['DeliveryDate'])
                
                collection_results[source_name] = {
                    'data': df,
                    'observations': len(df),
                    'success': True,
                    'priority': config['priority'],
                    'time_range': {
                        'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else 'Unknown',
                        'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else 'Unknown'
                    }
                }
                
                print(f"   ✅ {len(df):,} observations | {df['timestamp'].min() if 'timestamp' in df.columns else 'No timestamp'} to {df['timestamp'].max() if 'timestamp' in df.columns else ''}")
                
            else:
                collection_results[source_name] = {
                    'data': pd.DataFrame(),
                    'observations': 0,
                    'success': False,
                    'reason': 'No data returned'
                }
                print(f"   ⚠️  No data returned")
                
        except Exception as e:
            collection_results[source_name] = {
                'data': pd.DataFrame(),
                'observations': 0,
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Error: {e}")
    
    # Summary
    successful_sources = sum(1 for r in collection_results.values() if r.get('success', False))
    total_observations = sum(r.get('observations', 0) for r in collection_results.values())
    
    print(f"\n📊 Multi-source collection summary:")
    print(f"   Successful sources: {successful_sources}/{len(data_sources)}")
    print(f"   Total observations: {total_observations:,}")
    
    return collection_results


def get_primary_dataset(collection_results: dict) -> pd.DataFrame:
    """Extract primary dataset from multi-source collection results.
    
    Args:
        collection_results: Results from multi-source collection
    
    Returns:
        Primary dataset for archiving (Real-Time LMP with fallbacks)
    """
    # Try sources in priority order
    priority_order = ['real_time_lmp', 'dam_lmp', 'system_lambda']
    
    for source in priority_order:
        if source in collection_results and collection_results[source].get('success', False):
            df = collection_results[source]['data']
            if not df.empty:
                print(f"🎯 Using {source} as primary dataset ({len(df)} observations)")
                return df
    
    print("⚠️  No primary dataset available")
    return pd.DataFrame()

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


def archive_multi_source_data(collection_results: dict, dirs: dict) -> dict:
    """Archive data from multiple sources.
    
    Args:
        collection_results: Multi-source collection results
        dirs: Directory paths
    
    Returns:
        Archive statistics
    """
    stats = {'sources': {}}
    
    # Archive primary dataset (LMP data)
    primary_df = get_primary_dataset(collection_results)
    if not primary_df.empty:
        primary_stats = append_to_archive(primary_df, dirs['archive'])
        stats['primary'] = primary_stats
    
    # Archive ancillary data separately for risk modeling
    if 'ancillary_prices' in collection_results:
        ancillary_result = collection_results['ancillary_prices']
        if ancillary_result.get('success', False):
            ancillary_df = ancillary_result['data']
            
            print(f"\n💰 Archiving ancillary service prices...")
            
            if not ancillary_df.empty:
                try:
                    # Save ancillary data monthly
                    if 'timestamp' in ancillary_df.columns:
                        latest_date = ancillary_df['timestamp'].max()
                        ancillary_file = dirs['ancillary'] / f"ancillary_prices_{latest_date.year}_{latest_date.month:02d}.parquet"
                    else:
                        ancillary_file = dirs['ancillary'] / f"ancillary_prices_{datetime.now().strftime('%Y_%m')}.parquet"
                    
                    ancillary_df.to_parquet(ancillary_file, compression='snappy', index=False)
                    print(f"   ✅ Ancillary data saved: {ancillary_file.name}")
                    
                    stats['ancillary'] = {
                        'file': ancillary_file.name,
                        'records': len(ancillary_df)
                    }
                    
                except Exception as e:
                    print(f"   ⚠️  Ancillary archive failed: {e}")
                    stats['ancillary'] = {'error': str(e)}
    
    # Save multi-source summary
    multi_source_file = dirs['multi_source'] / f"daily_collection_{datetime.now().date().isoformat()}.json"
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'sources': {
            source: {
                'success': result.get('success', False),
                'observations': result.get('observations', 0),
                'time_range': result.get('time_range', {})
            }
            for source, result in collection_results.items()
        }
    }
    
    with open(multi_source_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    stats['multi_source_summary'] = multi_source_file.name
    
    return stats

def main():
    """Enhanced main execution with multi-source strategy."""
    try:
        # Setup
        dirs = setup_directories()
        client = ERCOTDataClient()
        
        # Multi-source data collection
        collection_results = fetch_daily_data_multi_source(client, max_reports=300)
        
        # Check if we got any data
        successful_sources = [s for s, r in collection_results.items() if r.get('success', False)]
        
        if not successful_sources:
            print("\n⚠️  No data collected from any source. Exiting.")
            return
        
        # Archive all collected data
        archive_stats = archive_multi_source_data(collection_results, dirs)
        
        # Generate report using primary dataset
        primary_df = get_primary_dataset(collection_results)
        if not primary_df.empty:
            report_stats = generate_daily_report(primary_df, dirs['reports'])
        else:
            report_stats = {'status': 'no_primary_data'}
        
        # Cleanup
        cleanup_cache(client, days=7)
        
        # Enhanced logging
        save_enhanced_log(dirs, {
            'collection_results': collection_results,
            'archive': archive_stats, 
            'report': report_stats
        })
        
        print(f"\n{'='*80}")
        print("✅ Enhanced daily collection complete!")
        print(f"   Sources used: {', '.join(successful_sources)}")
        print(f"   Primary dataset: {len(primary_df)} observations" if not primary_df.empty else "   No primary dataset")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error during enhanced daily collection: {e}")
        import traceback
        traceback.print_exc()
        raise

def save_enhanced_log(dirs: dict, stats: dict):
    """Save enhanced execution log with multi-source details.
    
    Args:
        dirs: Directory paths
        stats: Enhanced statistics from execution
    """
    log_file = dirs['logs'] / f"enhanced_collection_log_{datetime.now().date().isoformat()}.json"
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'status': 'success',
        'collection_summary': {
            source: {
                'success': result.get('success', False),
                'observations': result.get('observations', 0),
                'error': result.get('error') if 'error' in result else None
            }
            for source, result in stats.get('collection_results', {}).items()
        },
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
    
    print(f"\n📝 Enhanced execution log saved: {log_file.name}")


if __name__ == '__main__':
    main()
