"""
Main Ingestion Pipeline Orchestrator
Coordinates all data ingestion tasks
"""
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from loguru import logger

from .traffic_data import ingest_traffic_data
from .incidents import ingest_crash_data
from .weather import ingest_weather_data, generate_synthetic_weather
from .construction import ingest_construction_data
from ..config import get_settings

settings = get_settings()


def run_full_ingestion(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    database_url: Optional[str] = None,
    include_traffic: bool = True,
    include_crashes: bool = True,
    include_weather: bool = True,
    include_construction: bool = True,
    use_synthetic_weather: bool = False
) -> Dict[str, Any]:
    """
    Run complete data ingestion pipeline
    
    Args:
        start_date: Start date for historical data (default: 7 days ago)
        end_date: End date for data (default: now)
        database_url: PostgreSQL connection string
        include_traffic: Ingest traffic congestion data
        include_crashes: Ingest crash/incident data
        include_weather: Ingest weather data
        include_construction: Ingest road construction data
        use_synthetic_weather: Generate synthetic weather data (for testing)
    
    Returns:
        Dictionary with counts of ingested records
    """
    
    if database_url is None:
        database_url = settings.database_url
        
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now()
    
    logger.info("=" * 60)
    logger.info("Starting Urban Traffic Intelligence Data Ingestion")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("=" * 60)
    
    results = {
        'traffic_congestion': 0,
        'crashes': 0,
        'crash_documents': 0,
        'weather': 0,
        'construction': 0,
        'construction_documents': 0,
        'status': 'success',
        'errors': []
    }
    
    # 1. Ingest Traffic Congestion Data
    if include_traffic:
        try:
            logger.info("\n[1/4] Ingesting traffic congestion data...")
            results['traffic_congestion'] = ingest_traffic_data(
                start_date=start_date,
                end_date=end_date,
                database_url=database_url
            )
            logger.info(f"✓ Traffic data: {results['traffic_congestion']} records")
        except Exception as e:
            logger.error(f"✗ Traffic ingestion failed: {e}")
            results['errors'].append(f"traffic: {str(e)}")
    
    # 2. Ingest Crash/Incident Data
    if include_crashes:
        try:
            logger.info("\n[2/4] Ingesting crash/incident data...")
            crash_result = ingest_crash_data(
                start_date=start_date,
                end_date=end_date,
                database_url=database_url,
                create_docs=True
            )
            results['crashes'] = crash_result['crashes']
            results['crash_documents'] = crash_result['documents']
            logger.info(f"✓ Crash data: {results['crashes']} records, {results['crash_documents']} documents")
        except Exception as e:
            logger.error(f"✗ Crash ingestion failed: {e}")
            results['errors'].append(f"crashes: {str(e)}")
    
    # 3. Ingest Weather Data
    if include_weather:
        try:
            logger.info("\n[3/4] Ingesting weather data...")
            if use_synthetic_weather:
                results['weather'] = generate_synthetic_weather(
                    start_date=start_date,
                    end_date=end_date,
                    database_url=database_url
                )
                logger.info(f"✓ Synthetic weather: {results['weather']} records")
            else:
                results['weather'] = ingest_weather_data(
                    database_url=database_url,
                    include_forecast=True
                )
                logger.info(f"✓ Weather data: {results['weather']} records")
        except Exception as e:
            logger.error(f"✗ Weather ingestion failed: {e}")
            results['errors'].append(f"weather: {str(e)}")
    
    # 4. Ingest Construction Data
    if include_construction:
        try:
            logger.info("\n[4/4] Ingesting road construction data...")
            construction_result = ingest_construction_data(
                database_url=database_url,
                active_only=True,
                create_docs=True
            )
            results['construction'] = construction_result['construction']
            results['construction_documents'] = construction_result['documents']
            logger.info(f"✓ Construction: {results['construction']} records, {results['construction_documents']} documents")
        except Exception as e:
            logger.error(f"✗ Construction ingestion failed: {e}")
            results['errors'].append(f"construction: {str(e)}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Traffic Congestion: {results['traffic_congestion']:,} records")
    logger.info(f"Crashes: {results['crashes']:,} records")
    logger.info(f"Crash Documents: {results['crash_documents']:,} documents")
    logger.info(f"Weather: {results['weather']:,} records")
    logger.info(f"Construction: {results['construction']:,} records")
    logger.info(f"Construction Documents: {results['construction_documents']:,} documents")
    
    total_documents = results['crash_documents'] + results['construction_documents']
    logger.info(f"\nTotal RAG Documents: {total_documents:,}")
    
    if results['errors']:
        results['status'] = 'partial'
        logger.warning(f"\nErrors: {len(results['errors'])}")
        for error in results['errors']:
            logger.warning(f"  - {error}")
    
    return results


def main():
    """CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description='Urban Traffic Intelligence - Data Ingestion Pipeline'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD format)',
        default=None
    )
    parser.add_argument(
        '--end-date', 
        type=str,
        help='End date (YYYY-MM-DD format)',
        default=None
    )
    parser.add_argument(
        '--days',
        type=int,
        help='Number of days to look back (alternative to start-date)',
        default=7
    )
    parser.add_argument(
        '--database-url',
        type=str,
        help='PostgreSQL connection string',
        default=None
    )
    parser.add_argument(
        '--skip-traffic',
        action='store_true',
        help='Skip traffic congestion data'
    )
    parser.add_argument(
        '--skip-crashes',
        action='store_true',
        help='Skip crash/incident data'
    )
    parser.add_argument(
        '--skip-weather',
        action='store_true',
        help='Skip weather data'
    )
    parser.add_argument(
        '--skip-construction',
        action='store_true',
        help='Skip construction data'
    )
    parser.add_argument(
        '--synthetic-weather',
        action='store_true',
        help='Use synthetic weather data (for testing)'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    end_date = datetime.now()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.days)
    
    # Run ingestion
    results = run_full_ingestion(
        start_date=start_date,
        end_date=end_date,
        database_url=args.database_url,
        include_traffic=not args.skip_traffic,
        include_crashes=not args.skip_crashes,
        include_weather=not args.skip_weather,
        include_construction=not args.skip_construction,
        use_synthetic_weather=args.synthetic_weather
    )
    
    # Exit with error code if there were failures
    if results['errors']:
        exit(1)


if __name__ == "__main__":
    main()
