#!/usr/bin/env python3
"""CLI tool for ingesting historical ERCOT data.

This script fetches historical market data from ERCOT's public archives,
caches it locally, and provides options for data export and analysis.

Usage:
    python scripts/ingest_historical_data.py --report-type real_time_lmp --max-reports 10
    python scripts/ingest_historical_data.py --list-reports
    python scripts/ingest_historical_data.py --clear-cache
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hayeknet.data.client import ERCOTDataClient


def list_available_reports(client: ERCOTDataClient, report_type: str) -> None:
    """List available reports without downloading."""
    print(f"\n{'='*80}")
    print(f"Available {report_type} reports from ERCOT")
    print(f"{'='*80}\n")
    
    reports = client.fetch_historical_reports(report_type=report_type, use_cache=False)
    
    if not reports:
        print("❌ No reports found")
        return
    
    print(f"Found {len(reports)} reports (showing most recent 20):\n")
    
    for i, report in enumerate(reports[:20], 1):
        pub_date = report.get('PublishDate', 'Unknown')
        friendly_name = report.get('FriendlyName', 'Unknown')
        size_kb = int(report.get('ContentSize', 0)) / 1024
        doc_id = report.get('DocID', 'Unknown')
        
        print(f"{i:3d}. {pub_date:25s} | {friendly_name:50s} | {size_kb:6.1f} KB | DocID: {doc_id}")
    
    if len(reports) > 20:
        print(f"\n... and {len(reports) - 20} more reports")
    
    print(f"\nℹ️  Reports are kept in ERCOT archive for ~5 days")


def ingest_data(
    client: ERCOTDataClient,
    report_type: str,
    max_reports: int,
    start_date: datetime | None,
    end_date: datetime | None,
    export_csv: bool,
    use_cache: bool
) -> None:
    """Ingest historical data and optionally export."""
    print(f"\n{'='*80}")
    print(f"Ingesting {report_type} data from ERCOT")
    print(f"{'='*80}\n")
    
    print(f"📥 Configuration:")
    print(f"   - Report Type: {report_type}")
    print(f"   - Max Reports: {max_reports if max_reports else 'All available'}")
    print(f"   - Start Date: {start_date.date() if start_date else 'None'}")
    print(f"   - End Date: {end_date.date() if end_date else 'None'}")
    print(f"   - Use Cache: {use_cache}")
    print(f"   - Export CSV: {export_csv}\n")
    
    # Fetch data
    df = client.fetch_historical_data(
        report_type=report_type,
        start_date=start_date,
        end_date=end_date,
        max_reports=max_reports,
        use_cache=use_cache
    )
    
    if df.empty:
        print("\n❌ No data retrieved")
        return
    
    # Display summary statistics
    print(f"\n{'='*80}")
    print("Data Summary")
    print(f"{'='*80}\n")
    
    print(f"📊 Total Records: {len(df):,}")
    print(f"📊 Columns: {len(df.columns)}")
    print(f"📊 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if 'timestamp' in df.columns:
        print(f"\n⏰ Time Range:")
        print(f"   - Start: {df['timestamp'].min()}")
        print(f"   - End: {df['timestamp'].max()}")
        print(f"   - Duration: {df['timestamp'].max() - df['timestamp'].min()}")
    
    print(f"\n📋 Available Columns:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        print(f"   - {col:30s} | {str(dtype):15s} | {pct:5.1f}% non-null")
    
    # Show sample data
    print(f"\n📄 Sample Data (first 5 rows):")
    print(df.head().to_string())
    
    # Export if requested
    if export_csv:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = output_dir / f"{report_type}_{timestamp}.csv"
        parquet_file = output_dir / f"{report_type}_{timestamp}.parquet"
        
        df.to_csv(csv_file, index=False)
        df.to_parquet(parquet_file, compression='snappy')
        
        print(f"\n💾 Exported data:")
        print(f"   - CSV: {csv_file}")
        print(f"   - Parquet: {parquet_file}")


def clear_cache(client: ERCOTDataClient, older_than_days: int | None) -> None:
    """Clear cached data."""
    print(f"\n{'='*80}")
    print("Clearing Cache")
    print(f"{'='*80}\n")
    
    if older_than_days:
        print(f"🗑️  Clearing files older than {older_than_days} days...")
    else:
        print(f"🗑️  Clearing all cached files...")
    
    count = client.clear_cache(older_than_days=older_than_days)
    
    if count == 0:
        print("ℹ️  No files to clear")
    else:
        print(f"✅ Cleared {count} files")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest historical ERCOT market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available reports
  %(prog)s --list-reports --report-type real_time_lmp
  
  # Download latest 10 LMP reports
  %(prog)s --report-type real_time_lmp --max-reports 10
  
  # Download and export to CSV
  %(prog)s --report-type real_time_lmp --max-reports 5 --export-csv
  
  # Download with date range
  %(prog)s --report-type real_time_lmp --start-date 2025-09-28
  
  # Clear old cache files
  %(prog)s --clear-cache --older-than-days 7
  
  # Clear all cache
  %(prog)s --clear-cache

Available report types:
  - real_time_lmp: Real-time locational marginal prices (5-min intervals)
  - real_time_load: Real-time system load data
  - system_lambda: System lambda (marginal cost)
  - ancillary_prices: Ancillary service prices
  - dam_lmp: Day-ahead market LMPs
        """
    )
    
    # Action flags
    parser.add_argument(
        '--list-reports',
        action='store_true',
        help='List available reports without downloading'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cached data files'
    )
    
    # Data selection
    parser.add_argument(
        '--report-type',
        type=str,
        default='real_time_lmp',
        choices=['real_time_lmp', 'real_time_load', 'system_lambda', 'ancillary_prices', 'dam_lmp'],
        help='Type of report to fetch (default: real_time_lmp)'
    )
    parser.add_argument(
        '--max-reports',
        type=int,
        default=None,
        help='Maximum number of reports to download (default: all available)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for data range (format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for data range (format: YYYY-MM-DD)'
    )
    
    # Options
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export data to CSV and Parquet files'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching (always download fresh data)'
    )
    parser.add_argument(
        '--older-than-days',
        type=int,
        default=None,
        help='When clearing cache, only clear files older than N days'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Initialize client
    print("\n🚀 Initializing ERCOT Data Client...")
    client = ERCOTDataClient()
    print(f"📁 Cache directory: {client.cache_dir}")
    
    # Execute requested action
    if args.list_reports:
        list_available_reports(client, args.report_type)
    elif args.clear_cache:
        clear_cache(client, args.older_than_days)
    else:
        # Default action: ingest data
        ingest_data(
            client=client,
            report_type=args.report_type,
            max_reports=args.max_reports,
            start_date=start_date,
            end_date=end_date,
            export_csv=args.export_csv,
            use_cache=not args.no_cache
        )
    
    print(f"\n{'='*80}")
    print("✅ Complete")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
