#!/usr/bin/env python3
"""
Minerva CLI - Unified command-line interface for the Minerva data analysis platform.

Usage:
    minerva ingest s3 --bucket BUCKET --prefix PREFIX --dest datasets
    minerva ingest api URL --dest datasets
    minerva analyze FILE [--target COLUMN]
    minerva serve [--port PORT] [--host HOST]
    minerva dashboard [--port PORT]
    minerva test [--quick]
    minerva info
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cmd_ingest(args):
    """Ingest data from S3 or API sources. (DEPRECATED)"""
    print("WARNING: This command is deprecated. Please use the 'Minerva Ops Center' (minerva admin) for data ingestion.")
    from legacy.Data_Intake.datalake_ingestion import main as ingest_main
    
    # Build argv for the ingestion module
    ingest_args = [args.source]
    if args.source == "s3":
        ingest_args.extend(["--bucket", args.bucket, "--prefix", args.prefix])
    else:  # api
        ingest_args.append(args.url)
    
    if args.dest:
        ingest_args.extend(["--dest", args.dest])
    
    # Call ingestion
    sys.argv = ["datalake_ingestion"] + ingest_args
    ingest_main()


def cmd_analyze(args):
    """Run analysis on a data file."""
    import pandas as pd
    from orchestration.orchestrate_workflow import run_workflow
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Load data based on file type
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
    elif suffix == ".json":
        df = pd.read_json(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        print(f"Error: Unsupported file type: {suffix}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {file_path.name}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Run analysis
    print("Running analysis...")
    result = run_workflow(df, target=args.target)
    
    # Display results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nRun ID: {result.get('run_id', 'N/A')}")
    print(f"Analysis Type: {result.get('analysis_type', 'N/A')}")
    
    if result.get("metrics"):
        print("\nMetrics:")
        for key, value in result["metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    if result.get("insights"):
        print("\nInsights:")
        for insight in result["insights"]:
            print(f"  • {insight}")
    
    if result.get("feature_importance"):
        print("\nTop Features:")
        sorted_features = sorted(
            result["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for name, importance in sorted_features:
            print(f"  {name}: {importance:.4f}")
    
    if result.get("error"):
        print(f"\nError: {result['error']}")
    
    if result.get("warnings"):
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"  ⚠ {warning}")
    
    return result


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    
    print(f"Starting Minerva API server on {args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_dashboard(args):
    """Start the Streamlit dashboard."""
    import subprocess
    
    dashboard_path = PROJECT_ROOT / "ui" / "dashboard.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
    ]
    
    if args.host:
        cmd.extend(["--server.address", args.host])
    
    print(f"Starting Minerva Dashboard on port {args.port}")
    print("Press Ctrl+C to stop")
    print()
    
    subprocess.run(cmd)


def cmd_test(args):
    """Run the test suite."""
    import subprocess
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if args.quick:
        # Run only fast tests
        cmd.extend(["-m", "not slow", "-q"])
    else:
        cmd.extend(["-v"])
    
    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing"])
    
    print("Running tests...")
    subprocess.run(cmd, cwd=PROJECT_ROOT)


def cmd_info(args):
    """Display system and configuration information."""
    print("=" * 60)
    print("MINERVA PLATFORM INFO")
    print("=" * 60)
    
    # Python info
    print(f"\nPython: {sys.version}")
    print(f"Project Root: {PROJECT_ROOT}")
    
    # Check key dependencies
    print("\nDependencies:")
    deps = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("fastapi", "fastapi"),
        ("streamlit", "streamlit"),
        ("llama-cpp", "llama_cpp"),
    ]
    for name, module in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ✗ {name}: not installed")
    
    # Check LLM availability
    print("\nLLM Status:")
    try:
        from llm_manager.llm_interface import is_llm_available
        if is_llm_available():
            print("  ✓ Local LLM available")
        else:
            print("  ✗ No LLM model found")
    except Exception as e:
        print(f"  ✗ LLM check failed: {e}")
    
    # Configuration
    print("\nConfiguration:")
    from config import (
        LOCAL_DATA_DIR,
        LOG_DIR,
        ENABLE_PROMETHEUS,
        MAX_REQUESTS_FREE,
        MAX_GB_FREE,
    )
    print(f"  LOCAL_DATA_DIR: {LOCAL_DATA_DIR}")
    print(f"  LOG_DIR: {LOG_DIR}")
    print(f"  ENABLE_PROMETHEUS: {ENABLE_PROMETHEUS}")
    print(f"  MAX_REQUESTS_FREE: {MAX_REQUESTS_FREE}")
    print(f"  MAX_GB_FREE: {MAX_GB_FREE}")


def cmd_admin(args):
    """Start the Ops Center (Admin UI)."""
    import subprocess
    
    dashboard_path = PROJECT_ROOT / "ops_center.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
    ]
    
    if args.host:
        cmd.extend(["--server.address", args.host])
    
    print(f"Starting Minerva Ops Center on port {args.port}")
    print("Press Ctrl+C to stop")
    print()
    
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        prog="minerva",
        description="Minerva Data Analysis Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  minerva ingest s3 --bucket my-bucket --prefix data/
  minerva ingest api https://example.com/data.csv
  minerva analyze data.csv --target sales
  minerva serve --port 8000
  minerva dashboard
  minerva test --quick
  minerva info
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from sources")
    ingest_subparsers = ingest_parser.add_subparsers(dest="source")
    
    # ingest s3
    s3_parser = ingest_subparsers.add_parser("s3", help="Ingest from S3")
    s3_parser.add_argument("--bucket", required=True, help="S3 bucket name")
    s3_parser.add_argument("--prefix", default="", help="S3 key prefix")
    s3_parser.add_argument("--dest", default="datasets", help="Destination directory")
    
    # ingest api
    api_parser = ingest_subparsers.add_parser("api", help="Ingest from API/URL")
    api_parser.add_argument("url", help="URL to fetch data from")
    api_parser.add_argument("--dest", default="datasets", help="Destination directory")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a data file")
    analyze_parser.add_argument("file", help="Path to data file (CSV, Excel, JSON, Parquet)")
    analyze_parser.add_argument("--target", "-t", help="Target column for prediction")
    analyze_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Start the Streamlit dashboard")
    dash_parser.add_argument("--port", "-p", type=int, default=8501, help="Port number")
    dash_parser.add_argument("--host", default=None, help="Host address")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run the test suite")
    test_parser.add_argument("--quick", "-q", action="store_true", help="Run only fast tests")
    test_parser.add_argument("--coverage", "-c", action="store_true", help="Include coverage report")
    
    # Info command
    subparsers.add_parser("info", help="Display system information")

    # Admin command
    admin_parser = subparsers.add_parser("admin", help="Start the Ops Center (Admin UI)")
    admin_parser.add_argument("--port", "-p", type=int, default=8502, help="Port number")
    admin_parser.add_argument("--host", default=None, help="Host address")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Dispatch to command handler
    commands = {
        "ingest": cmd_ingest,
        "analyze": cmd_analyze,
        "serve": cmd_serve,
        "dashboard": cmd_dashboard,
        "test": cmd_test,
        "info": cmd_info,
        "admin": cmd_admin,
    }
    
    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
