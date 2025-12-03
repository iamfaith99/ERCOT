"""Data collection workflow functions."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from hayeknet.data.client import ERCOTDataClient


def fetch_daily_data(client: ERCOTDataClient, quick: bool = False, force_fresh: bool = False) -> pd.DataFrame:
    """Fetch latest ERCOT data.
    
    Args:
        client: ERCOTDataClient instance
        quick: If True, fetch only 1 hour of data (12 reports) instead of all available reports
        force_fresh: If True, bypass cache and download fresh data
    
    Returns:
        DataFrame with latest ERCOT LMP data
    """
    print(f"\n{'='*80}")
    print(f"STEP 1: Data Ingestion - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Limit to reasonable number to prevent memory issues
    # ERCOT publishes ~288 reports/day (every 5 min), so 500 = ~1.7 days of data
    max_reports = 12 if quick else 500  # 1 hour vs ~1.7 days
    
    print(f"📥 Fetching latest ERCOT LMP data...")
    print(f"   Mode: {'Quick test (1 hour)' if quick else 'Full (~1.7 days)'}")
    print(f"   Reports: ~{max_reports}")
    
    # Smart caching strategy:
    # - Use cache for individual report files (efficient, avoids re-downloading)
    # - BUT force fresh report LIST to get latest available reports
    # - This ensures we always get today's newest data while still leveraging cache
    
    today = datetime.now().date()
    cache_dir = client.cache_dir
    today_str = today.strftime('%Y%m%d')
    
    print(f"   📅 Date: {today}")
    
    # Always clear old report list cache to force fresh report discovery
    report_list_files = list(cache_dir.glob('report_list_*.json'))
    old_lists_cleared = 0
    for f in report_list_files:
        try:
            # Extract date from filename (e.g., 'report_list_real_time_lmp_2025-10-03.json')
            file_date_str = f.stem.split('_')[-1]  # e.g., '2025-10-03'
            if file_date_str != str(today):
                f.unlink()
                old_lists_cleared += 1
        except Exception:
            pass
    
    if old_lists_cleared > 0:
        print(f"   🗑️  Cleared {old_lists_cleared} old report list(s)")
    
    # Check what data we already have from today
    existing_files = list(cache_dir.glob(f'*{today_str}*.pkl'))
    
    if force_fresh:
        print(f"   🔄 Force fresh mode: will download new reports even if cached")
        use_cache = False
    else:
        if existing_files:
            # Get the newest cached file to check how recent it is
            newest_file = max(existing_files, key=lambda p: p.stat().st_mtime)
            cache_age_minutes = (datetime.now().timestamp() - newest_file.stat().st_mtime) / 60
            
            print(f"   📂 Found {len(existing_files)} cached reports from today")
            print(f"   🕒 Most recent cache: {cache_age_minutes:.1f} minutes ago")
            
            # If cache is recent (< 15 minutes), use it to avoid hammering ERCOT API
            if cache_age_minutes < 15:
                print(f"   ✅ Using cached data (< 15 min old)")
                use_cache = True
            else:
                print(f"   🔄 Cache is stale, fetching fresh reports")
                use_cache = False  # Will still use cached individual files, but get new report list
        else:
            print(f"   🆕 No cached data from today, fetching fresh reports")
            use_cache = False
    
    df = client.fetch_historical_data(
        report_type="real_time_lmp",
        max_reports=max_reports,
        use_cache=use_cache
    )
    
    if df.empty:
        print("⚠️  No new data available")
        return df
    
    # Parse timestamp - standardize column name
    if 'SCEDTimestamp' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['SCEDTimestamp'])
    elif 'DeliveryDate' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['DeliveryDate'])
        if 'DeliveryHour' in df.columns:
            df['timestamp'] += pd.to_timedelta((df['DeliveryHour'] - 1) * 60, unit='min')
        if 'DeliveryInterval' in df.columns:
            df['timestamp'] += pd.to_timedelta((df['DeliveryInterval'] - 1) * 5, unit='min')
    
    # Standardize LMP column name
    if 'LMP' in df.columns and 'lmp_usd' not in df.columns:
        df['lmp_usd'] = df['LMP']
    elif 'Settlement Point Price' in df.columns and 'lmp_usd' not in df.columns:
        df['lmp_usd'] = df['Settlement Point Price']
    
    print(f"\n✅ Data ingestion complete:")
    print(f"   Observations: {len(df):,}")
    if 'timestamp' in df.columns:
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    if 'settlement_point' in df.columns:
        print(f"   Settlement points: {df['settlement_point'].nunique():,}")
    
    return df

