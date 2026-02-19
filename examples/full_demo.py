#!/usr/bin/env python
"""Assay Full Feature Demo.

This script demonstrates all major features of the Assay platform:
1. Public data source registry and semantic index
2. Data upload and automatic preprocessing
3. Semantic enrichment with public datasets
4. Automatic analyzer selection and modeling
5. SHAP explanations and feature importance
6. Data quality scoring and diagnostics
7. LLM-powered insights (when available)
8. Dashboard visualization

Run with: python -m examples.full_demo
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Set environment for demo
os.environ.setdefault("LOCAL_DATA_DIR", "local_data")
os.environ.setdefault("LOG_DIR", "logs")


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step: int, description: str):
    """Print a step indicator."""
    print(f"\n{'─' * 60}")
    print(f"  Step {step}: {description}")
    print(f"{'─' * 60}")


def demo_public_data_registry():
    """Demonstrate the public data sources registry."""
    print_header("PUBLIC DATA SOURCES REGISTRY")
    
    from catalog.public_data_sources import (
        list_sources,
        populate_registry,
        setup_all_datasets,
        print_registry_summary,
    )
    
    print("\n📚 Initializing public data registry...")
    populate_registry()
    
    print("\n📊 Available data sources by category:")
    sources = list_sources()
    
    by_category = {}
    for src in sources:
        cat = src["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(src)
    
    for category, items in sorted(by_category.items()):
        print(f"\n  📁 {category.upper()}")
        for item in items:
            status = "✅" if item["is_available"] else "⬜"
            print(f"     {status} {item['name']}")
    
    print("\n🔧 Setting up datasets...")
    results = setup_all_datasets()
    available = sum(1 for v in results.values() if v)
    print(f"   {available}/{len(results)} datasets ready")
    
    return sources


def demo_semantic_index():
    """Demonstrate the semantic index."""
    print_header("SEMANTIC INDEX")
    
    from catalog.semantic_index import build_index, find_tables_by_roles, get_table_metadata
    from catalog.public_data_sources import DATASETS_DIR
    
    print("\n🔨 Building semantic index from datasets...")
    build_index(datasets_dir=str(DATASETS_DIR))
    
    # Show what roles are indexed
    test_roles = ["zip_code", "city", "state", "transaction_date", "store_id"]
    print(f"\n🔍 Finding tables with roles: {test_roles}")
    
    for role in test_roles:
        tables = find_tables_by_roles([role])
        if tables:
            print(f"   • {role}: {', '.join(tables[:3])}")
    
    # Show metadata for a sample table
    print("\n📋 Sample table metadata (stores.csv):")
    meta = get_table_metadata("stores.csv")
    for col, role in meta[:5]:
        print(f"   • {col} → {role}")


def demo_data_preprocessing():
    """Demonstrate data preprocessing and quality scoring."""
    print_header("DATA PREPROCESSING & QUALITY")
    
    from orchestration.data_preprocessor import DataPreprocessor
    from orchestration.data_quality_scorer import compute_safety_metrics, summarize_for_display
    
    # Load sample data
    sample_path = PROJECT_ROOT / "datasets" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not sample_path.exists():
        print("❌ Sample dataset not found")
        return None
    
    print(f"\n📂 Loading: {sample_path.name}")
    df = pd.read_csv(sample_path)
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Compute quality metrics
    print("\n📊 Computing data quality metrics...")
    metrics = compute_safety_metrics(df)
    summary = summarize_for_display(metrics)
    
    print(f"\n   {summary['status_emoji']} Quality Score: {summary['score']}%")
    print(f"   📈 Completeness: {summary['completeness_pct']}%")
    print(f"   📋 Complete Rows: {summary['rows_complete_pct']}%")
    
    if summary['warnings']:
        print("\n   ⚠️ Warnings:")
        for w in summary['warnings'][:3]:
            print(f"      • {w}")
    
    if summary['missing_columns']:
        print(f"\n   📝 Columns with missing data: {len(summary['missing_columns'])}")
    
    return df


def demo_column_role_inference():
    """Demonstrate column role inference."""
    print_header("COLUMN ROLE INFERENCE")
    
    from preprocessing.metadata_parser import infer_column_meta
    
    # Load sample data
    sample_path = PROJECT_ROOT / "datasets" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(sample_path)
    
    print("\n🏷️ Inferring column roles...")
    meta = infer_column_meta(df)
    
    print("\n   Column → Role (Confidence)")
    print("   " + "-" * 40)
    for m in meta[:10]:
        conf = f"{m.confidence:.0%}" if m.confidence else "?"
        print(f"   {m.name[:20]:<20} → {m.role:<15} ({conf})")
    
    if len(meta) > 10:
        print(f"   ... and {len(meta) - 10} more columns")
    
    return meta


def demo_semantic_enrichment():
    """Demonstrate semantic enrichment."""
    print_header("SEMANTIC ENRICHMENT")
    
    from Integration.semantic_merge import rank_and_merge, find_candidate_tables
    from preprocessing.metadata_parser import infer_column_meta
    
    # Create a simple user dataset with clear semantic columns
    print("\n📝 Creating sample user dataset...")
    user_data = pd.DataFrame({
        "transaction_date": pd.date_range("2023-01-01", periods=30, freq="D"),
        "store_id": [1, 2, 3, 1, 2, 3] * 5,
        "revenue": [100, 150, 200, 120, 180, 220] * 5,
        "city": ["New York", "Los Angeles", "Chicago"] * 10,
        "state": ["NY", "CA", "IL"] * 10,
    })
    print(f"   Shape: {user_data.shape}")
    print(f"   Columns: {list(user_data.columns)}")
    
    # Infer roles
    meta = infer_column_meta(user_data)
    print("\n   Inferred roles:")
    for m in meta:
        print(f"      {m.name} → {m.role}")
    
    print("\n🔍 Finding candidate enrichment tables...")
    candidates = find_candidate_tables(meta)
    print(f"   Found {len(candidates)} candidate tables: {candidates[:5]}")
    
    if not candidates:
        print("   No candidates found - this is expected if semantic index needs rebuilding")
        print("   Run: python -m catalog.public_data_sources rebuild")
        return user_data, {"tables_considered": [], "chosen_table": None}
    
    # Perform enrichment (tries multiple tables, skips failures)
    print("\n🔗 Performing semantic merge (brute-force approach)...")
    enriched_df, report = rank_and_merge(user_data, meta, max_merges=3)
    
    print(f"\n   📊 Merge Summary:")
    print(f"      Tables considered: {report.get('tables_considered', 0)}")
    print(f"      Successful merges: {report.get('successful_merges', 0)}")
    print(f"      Failed attempts: {report.get('failed_attempts', 0)}")
    print(f"      Total new columns: {report.get('total_new_columns', 0)}")
    
    if report.get('merged_tables'):
        print(f"\n   ✅ Successfully merged with:")
        for detail in report.get('details', []):
            if detail.get('joined'):
                print(f"      • {detail['table']} (+{detail['new_columns']} cols, {detail.get('match_rate', '?')} match)")
    
    # Show failed attempts (for transparency)
    failed = [d for d in report.get('details', []) if not d.get('joined')]
    if failed:
        print(f"\n   ⚠️ Failed attempts ({len(failed)}):")
        for f in failed[:3]:  # Show first 3
            print(f"      • {f['table']}: {f.get('reason', 'unknown')}")
        if len(failed) > 3:
            print(f"      ... and {len(failed) - 3} more")
    
    print(f"\n   Original shape: {user_data.shape}")
    print(f"   Enriched shape: {enriched_df.shape}")
    
    return enriched_df, report


def demo_analyzer_selection():
    """Demonstrate automatic analyzer selection."""
    print_header("AUTOMATIC ANALYZER SELECTION")
    
    from orchestration.analysis_selector import select_analyzer
    from modeling import REGISTRY
    
    print("\n📋 Available analyzers:")
    for name, analyzer_cls in REGISTRY.items():
        print(f"   • {name}")
    
    # Load sample data
    sample_path = PROJECT_ROOT / "datasets" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(sample_path)
    
    print(f"\n🎯 Selecting best analyzer for Telco Churn dataset...")
    analyzer = select_analyzer(df)
    print(f"   Selected: {analyzer.__class__.__name__}")
    
    return analyzer


def demo_full_workflow():
    """Demonstrate the full workflow."""
    print_header("FULL WORKFLOW EXECUTION")
    
    from orchestration.orchestrate_workflow import orchestrate_workflow
    from storage.local_backend import load_datalake_dfs
    
    print("\n🚀 Running full Assay workflow...")
    print("   This demonstrates:")
    print("   • Data loading and preprocessing")
    print("   • Semantic enrichment")
    print("   • Automatic model selection")
    print("   • Training and evaluation")
    print("   • Output generation")
    
    # Prepare inputs
    user_id = "demo_user"
    file_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Copy file to User_Data if needed
    src = PROJECT_ROOT / "datasets" / file_name
    dst = PROJECT_ROOT / "User_Data" / user_id / file_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and src.exists():
        import shutil
        shutil.copy(src, dst)
    
    print(f"\n   User: {user_id}")
    print(f"   File: {file_name}")
    
    # Load datalake
    datalake_dfs = load_datalake_dfs()
    print(f"   Datalake tables: {len(datalake_dfs)}")
    
    # Run workflow with diagnostics enabled
    print("\n   ⏳ Executing workflow (this may take a moment)...")
    
    try:
        result = orchestrate_workflow(
            user_id=user_id,
            file_name=file_name,
            datalake_dfs=datalake_dfs,
            target_column="Churn",
            diagnostics_config={
                "check_misalignment": True,
                "score_imputations": True,
                "monitor_drift": True,
            }
        )
        
        print("\n   ✅ Workflow completed!")
        
        # Show results
        if result:
            print(f"\n   📊 Results:")
            if "analysis_type" in result:
                print(f"      Analysis type: {result['analysis_type']}")
            if "metrics" in result:
                metrics = result["metrics"]
                for k, v in list(metrics.items())[:5]:
                    if isinstance(v, float):
                        print(f"      {k}: {v:.4f}")
            if "run_id" in result:
                print(f"      Run ID: {result['run_id']}")
            if result.get("needs_role_review"):
                print("      ⚠️ Model suggests reviewing column roles")
            if result.get("diagnostics"):
                print(f"      📋 Diagnostics available: {list(result['diagnostics'].keys())}")
        
        return result
        
    except Exception as e:
        print(f"\n   ❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_shap_explanations():
    """Demonstrate SHAP explanations."""
    print_header("MODEL EXPLANATIONS (SHAP)")
    
    from config import ENABLE_SHAP_EXPLANATIONS
    
    print(f"\n   SHAP Explanations enabled: {ENABLE_SHAP_EXPLANATIONS}")
    
    if ENABLE_SHAP_EXPLANATIONS:
        print("\n   When models are trained, SHAP values are computed to explain")
        print("   which features contribute most to predictions.")
        print("\n   Example output structure:")
        print("   {")
        print('     "feature_importances": [0.25, 0.18, 0.15, ...],')
        print('     "shap_values": [[0.1, -0.2, ...], ...],')
        print('     "shap_importance": [0.22, 0.19, 0.12, ...]')
        print("   }")
    else:
        print("\n   Set ENABLE_SHAP_EXPLANATIONS=true to enable")


def demo_llm_features():
    """Demonstrate LLM features."""
    print_header("LLM FEATURES")
    
    from config import ENABLE_LOCAL_LLM
    
    print(f"\n   Local LLM enabled: {ENABLE_LOCAL_LLM}")
    
    print("\n   LLM-powered features include:")
    print("   • Column role inference enhancement")
    print("   • Model recommendations")
    print("   • Business summary generation")
    print("   • Visualization selection (when heuristics uncertain)")
    print("   • Chatbot responses")
    
    if ENABLE_LOCAL_LLM:
        print("\n   Testing LLM availability...")
        try:
            from preprocessing.llm_preprocessor import recommend_models_with_llm
            sample_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            result = recommend_models_with_llm(sample_df)
            if "LLM unavailable" in result:
                print("   ⚠️ LLM model not loaded (download required)")
            else:
                print("   ✅ LLM is operational")
                print(f"   Sample recommendation: {result[:100]}...")
        except Exception as e:
            print(f"   ❌ LLM error: {e}")
    else:
        print("\n   To enable LLM features:")
        print("   1. Set ENABLE_LOCAL_LLM=true")
        print("   2. Set AUTO_DOWNLOAD_LLM=true (to auto-download model)")
        print("   3. Or manually download a GGUF model and set MISTRAL_MODEL_PATH")


def demo_visualization():
    """Demonstrate visualization selection."""
    print_header("VISUALIZATION SELECTION")
    
    from scripts.visualization_selector import select_visualization
    
    print("\n   Assay uses a hybrid approach:")
    print("   • Heuristics for high-confidence cases (>70%)")
    print("   • LLM fallback for uncertain cases")
    
    # Test cases
    test_cases = [
        {"columns": ["date", "sales"], "question": "Show sales over time"},
        {"columns": ["category", "revenue"], "question": "Compare revenue by category"},
        {"columns": ["x", "y"], "question": "Show correlation"},
    ]
    
    print("\n   Example selections:")
    for case in test_cases:
        try:
            result = select_visualization(
                case["columns"],
                case["question"],
                model_type="regression"
            )
            print(f"   • {case['question'][:30]}... → {result.get('chart_type', 'unknown')}")
        except Exception as e:
            print(f"   • {case['question'][:30]}... → error: {e}")


def print_demo_commands():
    """Print commands for running the demo."""
    print_header("DEMO COMMANDS")
    
    print("""
   To run individual components:
   
   # List public data sources
   python -m catalog.public_data_sources list
   
   # Setup all datasets
   python -m catalog.public_data_sources setup
   
   # Rebuild semantic index
   python -m catalog.public_data_sources rebuild
   
   # Run health tests
   pytest tests/test_health.py -v
   
   # Start API server
   uvicorn main:app --reload
   
   # Start dashboard
   streamlit run ui/dashboard.py
   
   # Run with LLM enabled
   set ENABLE_LOCAL_LLM=true
   set AUTO_DOWNLOAD_LLM=true
   python main.py
   
   # Run in DEV_MODE (no LLM, reduced limits)
   set DEV_MODE=true
   python main.py
""")


def main():
    """Run the full demo."""
    print("\n" + "🌟" * 35)
    print("         ASSAY FULL FEATURE DEMO")
    print("🌟" * 35)
    
    print(f"\nDemo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    
    try:
        # Step 1: Public Data Registry
        print_step(1, "Public Data Sources Registry")
        demo_public_data_registry()
        
        # Step 2: Semantic Index
        print_step(2, "Semantic Index")
        demo_semantic_index()
        
        # Step 3: Data Preprocessing
        print_step(3, "Data Preprocessing & Quality")
        demo_data_preprocessing()
        
        # Step 4: Column Role Inference
        print_step(4, "Column Role Inference")
        demo_column_role_inference()
        
        # Step 5: Semantic Enrichment
        print_step(5, "Semantic Enrichment")
        demo_semantic_enrichment()
        
        # Step 6: Analyzer Selection
        print_step(6, "Automatic Analyzer Selection")
        demo_analyzer_selection()
        
        # Step 7: SHAP Explanations
        print_step(7, "Model Explanations (SHAP)")
        demo_shap_explanations()
        
        # Step 8: LLM Features
        print_step(8, "LLM Features")
        demo_llm_features()
        
        # Step 9: Visualization
        print_step(9, "Visualization Selection")
        demo_visualization()
        
        # Step 10: Full Workflow (optional - takes time)
        print_step(10, "Full Workflow Execution")
        response = input("\n   Run full workflow? (y/n): ").strip().lower()
        if response == 'y':
            demo_full_workflow()
        else:
            print("   Skipped (run with 'y' to execute)")
        
        # Print commands
        print_demo_commands()
        
        print_header("DEMO COMPLETE")
        print("\n   ✅ All demo steps completed successfully!")
        print("\n   Next steps:")
        print("   1. Start the dashboard: streamlit run ui/dashboard.py")
        print("   2. Upload a CSV file and watch the magic happen")
        print("   3. Check the Data Quality Report and Model Explanations")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n   Demo interrupted by user")
    except Exception as e:
        print(f"\n\n   ❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
