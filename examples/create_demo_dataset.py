"""Sample retail sales dataset for demonstrating semantic enrichment.

This dataset simulates a retail store's transaction data with geographic
identifiers that can be enriched with demographic data.
"""

import pandas as pd
from pathlib import Path

# Create a realistic retail sales dataset
data = {
    "transaction_id": list(range(1001, 1101)),
    "transaction_date": pd.date_range("2024-01-01", periods=100, freq="D"),
    "store_id": ["S001", "S002", "S003", "S001", "S002"] * 20,
    "product_category": ["Electronics", "Clothing", "Groceries", "Home & Garden", "Sports"] * 20,
    "product_id": [f"P{i:04d}" for i in range(100)],
    "quantity": [1, 2, 1, 3, 2, 1, 4, 1, 2, 1] * 10,
    "unit_price": [
        299.99, 49.99, 12.99, 89.99, 39.99,
        599.99, 29.99, 8.99, 149.99, 59.99
    ] * 10,
    "customer_id": [f"C{i:05d}" for i in range(100)],
    "payment_method": ["Credit Card", "Debit Card", "Cash", "Credit Card", "Digital Wallet"] * 20,
    "zip_code": [
        "10001", "90210", "60601", "33101", "98101",
        "02101", "75201", "85001", "19101", "30301"
    ] * 10,
    "city": [
        "New York", "Beverly Hills", "Chicago", "Miami", "Seattle",
        "Boston", "Dallas", "Phoenix", "Philadelphia", "Atlanta"
    ] * 10,
    "state": [
        "NY", "CA", "IL", "FL", "WA",
        "MA", "TX", "AZ", "PA", "GA"
    ] * 10,
}

# Calculate total
df = pd.DataFrame(data)
df["total_price"] = df["quantity"] * df["unit_price"]
df["total_price"] = df["total_price"].round(2)

# Add some realistic variation
import random
random.seed(42)
df["discount"] = [round(random.choice([0, 0, 0, 0.1, 0.15, 0.2]) * 100, 0) for _ in range(100)]
df["online_order"] = [random.choice([True, False]) for _ in range(100)]

# Save to datasets folder
output_path = Path(__file__).parent.parent / "datasets" / "demo_retail_sales.csv"
df.to_csv(output_path, index=False)
print(f"Created demo dataset: {output_path}")
