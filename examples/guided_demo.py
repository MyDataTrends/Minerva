"""
Guided Demo Script - Quick Start tutorial for Minerva.

Run this script to see Minerva's key features in action:
1. Load demo retail sales data
2. Demonstrate semantic enrichment with census data  
3. Show automatic analysis selection
4. Generate an HTML report

Usage:
    python -m examples.guided_demo
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step_num: int, text: str):
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 40)


def run_demo():
    """Run the guided demo showcasing Minerva's key features."""
    
    print_header("🔮 Minerva Platform - Guided Demo")
    print("This demo showcases the key differentiating features of Minerva:\n")
    print("  1. Semantic data enrichment")
    print("  2. Automatic analysis selection")
    print("  3. LLM-powered insights (if available)")
    print("  4. Professional report generation")
    
    # Step 1: Load demo data
    print_step(1, "Loading Demo Retail Sales Data")
    
    datasets_dir = project_root / "datasets"
    demo_file = datasets_dir / "demo_retail_sales.csv"
    
    if not demo_file.exists():
        print(f"❌ Demo file not found: {demo_file}")
        print("   Creating sample data...")
        # Create inline if needed
        df = pd.DataFrame({
            "transaction_date": pd.date_range("2024-01-01", periods=50, freq="D"),
            "store_id": ["S001", "S002", "S003"] * 17 + ["S001"],
            "zip_code": ["10001", "90210", "60601", "33101", "98101"] * 10,
            "product_category": ["Electronics", "Clothing", "Groceries"] * 17 + ["Electronics"],
            "quantity": [1, 2, 1, 3, 2] * 10,
            "unit_price": [299.99, 49.99, 12.99, 89.99, 39.99] * 10,
            "city": ["New York", "Beverly Hills", "Chicago", "Miami", "Seattle"] * 10,
            "state": ["NY", "CA", "IL", "FL", "WA"] * 10,
        })
        df["total_price"] = df["quantity"] * df["unit_price"]
    else:
        df = pd.read_csv(demo_file)
    
    print(f"✅ Loaded {len(df)} transactions with {len(df.columns)} columns")
    print(f"   Columns: {', '.join(df.columns[:6])}...")
    print(f"\nPreview:")
    print(df.head(3).to_string(index=False))
    
    # Step 2: Semantic column detection
    print_step(2, "Detecting Semantic Column Roles")
    
    try:
        from preprocessing.metadata_parser import infer_column_meta
        meta = infer_column_meta(df)
        
        print("Detected roles:")
        for m in meta:
            if m.role != "unknown":
                print(f"   • {m.name}: {m.role}")
    except Exception as e:
        print(f"⚠️ Role detection unavailable: {e}")
        meta = []
    
    # Step 3: Semantic enrichment demo
    print_step(3, "Demonstrating Semantic Data Enrichment")
    
    census_file = datasets_dir / "census_zip_income.csv"
    if census_file.exists():
        census_df = pd.read_csv(census_file)
        print(f"📊 Census data available with columns: {', '.join(census_df.columns)}")
        
        # Show how semantic merge would work
        try:
            from Integration.semantic_merge import synthesise_join_keys
            
            # Demonstrate join key synthesis
            if meta:
                table_meta = [("zip_code", "zip_code"), ("median_income", "income")]
                u_df, t_df, join_keys, method = synthesise_join_keys(df, census_df, meta, table_meta)
                
                if join_keys:
                    print(f"\n✅ Automatic join key detected: {join_keys}")
                    print(f"   Method: {method}")
                    
                    # Perform actual merge
                    df["zip_code"] = df["zip_code"].astype(str)
                    census_df["zip_code"] = census_df["zip_code"].astype(str)
                    enriched = df.merge(census_df, on="zip_code", how="left")
                    new_cols = set(enriched.columns) - set(df.columns)
                    print(f"\n🔗 Enrichment would add {len(new_cols)} columns:")
                    for col in new_cols:
                        print(f"   + {col}")
                else:
                    print("⚠️ No automatic join keys found")
        except Exception as e:
            print(f"⚠️ Semantic merge demo unavailable: {e}")
    else:
        print("⚠️ Census data not found for enrichment demo")
    
    # Step 4: Analysis suggestion
    print_step(4, "Automatic Analysis Selection")
    
    try:
        from orchestration.analysis_selector import select_analyzer
        analyzer = select_analyzer(df)
        print(f"✅ Recommended analyzer: {analyzer.__class__.__name__}")
        print(f"   Based on: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"⚠️ Analyzer selection unavailable: {e}")
    
    # Step 5: Report generation
    print_step(5, "Report Generation")
    
    try:
        from reports.report_generator import generate_html_report, export_report_to_file
        
        # Create a mock result for demo
        mock_result = {
            "analysis_type": "regression",
            "summary": "Demo analysis of retail sales data showing seasonal patterns and regional variations.",
            "model_info": {
                "metrics": {"r2_score": 0.85, "mae": 12.5},
                "explanations": {
                    "feature_importances": {"unit_price": 0.35, "quantity": 0.28, "product_category": 0.22}
                }
            },
            "diagnostics": {"missing_pct": 0.5, "duplicate_rows": 0}
        }
        
        # Generate report
        output_path = project_root / "output" / "demo_report.html"
        output_path.parent.mkdir(exist_ok=True)
        
        export_report_to_file(df, mock_result, str(output_path), title="Minerva Demo Report")
        print(f"✅ Report generated: {output_path}")
        print(f"   Open this file in a browser to see the styled report")
    except Exception as e:
        print(f"⚠️ Report generation unavailable: {e}")
    
    # Summary
    print_header("Demo Complete!")
    print("Key Minerva differentiators demonstrated:")
    print("\n  🧠 Semantic Column Detection")
    print("     Automatically identifies column meanings (zip_code, sales_amount, etc.)")
    print("\n  🔗 Semantic Data Enrichment")
    print("     Enriches user data with public datasets using smart join key synthesis")
    print("\n  🎯 Automatic Analysis Selection")
    print("     Chooses the best ML approach based on data characteristics")
    print("\n  📄 Professional Report Export")
    print("     Generates styled HTML reports with metrics and insights")
    print("\n" + "="*60)
    print("To explore interactively, run: streamlit run ui/dashboard.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_demo()
