"""Public Data Sources Registry.

This module provides a curated database of public datasets that can be used
for semantic enrichment. Each dataset includes metadata about its columns,
roles, and how to download/access it.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.request import urlretrieve
import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "public_data_registry.db"
DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"


@dataclass
class PublicDataSource:
    """Metadata for a public data source."""
    name: str
    description: str
    category: str  # retail, demographics, weather, economic, geographic, holidays
    source_url: str
    local_filename: str
    columns: List[Dict[str, str]]  # [{"name": "col", "role": "zip_code", "description": "..."}]
    license: str = "Public Domain"
    update_frequency: str = "static"  # static, daily, weekly, monthly, yearly
    row_count: Optional[int] = None
    file_size_kb: Optional[int] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# ============================================================================
# CURATED PUBLIC DATA SOURCES
# ============================================================================

PUBLIC_DATA_SOURCES: List[PublicDataSource] = [
    # --- DEMOGRAPHICS ---
    PublicDataSource(
        name="US Census ZIP Code Income",
        description="Median household income by ZIP code from US Census Bureau ACS data",
        category="demographics",
        source_url="bundled",  # Already in datasets/
        local_filename="census_zip_income.csv",
        columns=[
            {"name": "zip_code", "role": "zip_code", "description": "5-digit ZIP code"},
            {"name": "median_income", "role": "income", "description": "Median household income"},
            {"name": "population", "role": "population", "description": "Total population"},
            {"name": "state", "role": "state", "description": "State abbreviation"},
        ],
        license="Public Domain (US Census)",
        tags=["income", "census", "demographics", "zip"],
    ),
    
    # --- HOLIDAYS ---
    PublicDataSource(
        name="US Federal Holidays",
        description="US federal holidays with dates and types",
        category="holidays",
        source_url="bundled",
        local_filename="holidays.csv",
        columns=[
            {"name": "date", "role": "holiday_date", "description": "Holiday date"},
            {"name": "holiday", "role": "holiday_name", "description": "Holiday name"},
            {"name": "type", "role": "holiday_type", "description": "Type of holiday"},
        ],
        license="Public Domain",
        tags=["holidays", "calendar", "dates"],
    ),
    
    PublicDataSource(
        name="Ecuador Holidays & Events",
        description="Holidays and special events in Ecuador (for retail forecasting)",
        category="holidays",
        source_url="bundled",
        local_filename="holidays_events.csv",
        columns=[
            {"name": "date", "role": "holiday_date", "description": "Event date"},
            {"name": "type", "role": "event_type", "description": "Event type"},
            {"name": "locale", "role": "locale", "description": "Geographic scope"},
            {"name": "locale_name", "role": "locale_name", "description": "Location name"},
            {"name": "description", "role": "event_description", "description": "Event description"},
            {"name": "transferred", "role": "transferred_flag", "description": "If holiday was transferred"},
        ],
        license="Public Domain",
        tags=["holidays", "events", "ecuador", "retail"],
    ),
    
    # --- ECONOMIC ---
    PublicDataSource(
        name="Oil Prices Daily",
        description="Daily oil prices (Brent crude) for economic analysis",
        category="economic",
        source_url="bundled",
        local_filename="oil.csv",
        columns=[
            {"name": "date", "role": "transaction_date", "description": "Price date"},
            {"name": "dcoilwtico", "role": "oil_price", "description": "WTI crude oil price USD"},
        ],
        license="Public Domain",
        tags=["oil", "commodities", "economic", "prices"],
    ),
    
    # --- RETAIL ---
    PublicDataSource(
        name="Store Master Data",
        description="Store location and attribute master data template",
        category="retail",
        source_url="bundled",
        local_filename="stores.csv",
        columns=[
            {"name": "store_nbr", "role": "store_id", "description": "Store number"},
            {"name": "city", "role": "city", "description": "City name"},
            {"name": "state", "role": "state", "description": "State/province"},
            {"name": "type", "role": "store_type", "description": "Store type classification"},
            {"name": "cluster", "role": "store_cluster", "description": "Store cluster grouping"},
        ],
        license="Public Domain",
        tags=["stores", "retail", "locations"],
    ),
    
    PublicDataSource(
        name="Transaction History Sample",
        description="Sample retail transaction data for testing",
        category="retail",
        source_url="bundled",
        local_filename="transactions.csv",
        columns=[
            {"name": "date", "role": "transaction_date", "description": "Transaction date"},
            {"name": "store_nbr", "role": "store_id", "description": "Store number"},
            {"name": "transactions", "role": "transaction_count", "description": "Number of transactions"},
        ],
        license="Public Domain",
        tags=["transactions", "retail", "sales"],
    ),
    
    # --- WEATHER ---
    PublicDataSource(
        name="Weather Sample Data",
        description="Sample weather data with temperature and conditions",
        category="weather",
        source_url="bundled",
        local_filename="weather_sample.csv",
        columns=[
            {"name": "date", "role": "transaction_date", "description": "Weather date"},
            {"name": "city", "role": "city", "description": "City name"},
            {"name": "temperature", "role": "temperature", "description": "Temperature in Fahrenheit"},
            {"name": "precipitation", "role": "precipitation", "description": "Precipitation in inches"},
            {"name": "conditions", "role": "weather_conditions", "description": "Weather conditions"},
        ],
        license="Public Domain",
        tags=["weather", "temperature", "climate"],
    ),
    
    # --- TELECOM/CHURN ---
    PublicDataSource(
        name="Telco Customer Churn",
        description="Telecom customer data for churn prediction modeling",
        category="telecom",
        source_url="bundled",
        local_filename="WA_Fn-UseC_-Telco-Customer-Churn.csv",
        columns=[
            {"name": "customerID", "role": "customer_id", "description": "Unique customer identifier"},
            {"name": "gender", "role": "customer_gender", "description": "Customer gender"},
            {"name": "tenure", "role": "tenure", "description": "Months with company"},
            {"name": "MonthlyCharges", "role": "monthly_amount", "description": "Monthly charges"},
            {"name": "TotalCharges", "role": "total_amount", "description": "Total charges"},
            {"name": "Churn", "role": "churn_flag", "description": "Customer churned (Yes/No)"},
        ],
        license="IBM Sample Data",
        tags=["telecom", "churn", "classification", "customer"],
    ),
    
    # --- GEOGRAPHIC MAPPINGS ---
    PublicDataSource(
        name="US ZIP to FIPS Mapping",
        description="Mapping from ZIP codes to FIPS county codes",
        category="geographic",
        source_url="generate",  # Generated locally
        local_filename="zip_to_fips.csv",
        columns=[
            {"name": "zip", "role": "zip_code", "description": "5-digit ZIP code"},
            {"name": "fips", "role": "fips_code", "description": "5-digit FIPS county code"},
            {"name": "state", "role": "state", "description": "State abbreviation"},
            {"name": "county", "role": "county", "description": "County name"},
        ],
        license="Public Domain",
        tags=["geographic", "zip", "fips", "mapping"],
    ),
    
    PublicDataSource(
        name="US State Codes",
        description="US state names, abbreviations, and FIPS codes",
        category="geographic",
        source_url="generate",  # Will be generated
        local_filename="us_states.csv",
        columns=[
            {"name": "state_name", "role": "state_name", "description": "Full state name"},
            {"name": "state_abbr", "role": "state", "description": "2-letter abbreviation"},
            {"name": "state_fips", "role": "state_fips", "description": "2-digit state FIPS"},
            {"name": "region", "role": "region", "description": "Census region"},
        ],
        license="Public Domain",
        tags=["geographic", "states", "regions"],
    ),
]


# ============================================================================
# REGISTRY DATABASE OPERATIONS
# ============================================================================

def init_registry(db_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """Initialize the public data registry database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS data_sources (
            name TEXT PRIMARY KEY,
            description TEXT,
            category TEXT,
            source_url TEXT,
            local_filename TEXT,
            license TEXT,
            update_frequency TEXT,
            row_count INTEGER,
            file_size_kb INTEGER,
            tags TEXT,
            columns TEXT,
            is_available INTEGER DEFAULT 0
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS source_columns (
            source_name TEXT,
            column_name TEXT,
            role TEXT,
            description TEXT,
            PRIMARY KEY (source_name, column_name),
            FOREIGN KEY (source_name) REFERENCES data_sources(name)
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized registry at {db_path}")


def populate_registry(
    sources: List[PublicDataSource] = None,
    db_path: Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """Populate the registry with data source definitions."""
    sources = sources or PUBLIC_DATA_SOURCES
    init_registry(db_path)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    for src in sources:
        # Check if file exists locally
        local_path = DATASETS_DIR / src.local_filename
        is_available = 1 if local_path.exists() else 0
        
        cur.execute("""
            INSERT OR REPLACE INTO data_sources 
            (name, description, category, source_url, local_filename, license, 
             update_frequency, row_count, file_size_kb, tags, columns, is_available)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            src.name, src.description, src.category, src.source_url,
            src.local_filename, src.license, src.update_frequency,
            src.row_count, src.file_size_kb,
            json.dumps(src.tags), json.dumps(src.columns), is_available
        ))
        
        # Insert column metadata
        for col in src.columns:
            cur.execute("""
                INSERT OR REPLACE INTO source_columns 
                (source_name, column_name, role, description)
                VALUES (?, ?, ?, ?)
            """, (src.name, col["name"], col["role"], col.get("description", "")))
    
    conn.commit()
    conn.close()
    logger.info(f"Populated registry with {len(sources)} data sources")


def list_sources(
    category: Optional[str] = None,
    available_only: bool = False,
    db_path: Path = DEFAULT_REGISTRY_PATH,
) -> List[Dict[str, Any]]:
    """List all registered data sources."""
    if not db_path.exists():
        populate_registry(db_path=db_path)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    query = "SELECT * FROM data_sources WHERE 1=1"
    params = []
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    if available_only:
        query += " AND is_available = 1"
    
    cur.execute(query, params)
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]


def get_source_by_name(name: str, db_path: Path = DEFAULT_REGISTRY_PATH) -> Optional[Dict]:
    """Get a specific data source by name."""
    if not db_path.exists():
        populate_registry(db_path=db_path)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM data_sources WHERE name = ?", (name,))
    columns = [desc[0] for desc in cur.description]
    row = cur.fetchone()
    conn.close()
    
    if row:
        return dict(zip(columns, row))
    return None


def find_sources_by_role(
    roles: List[str],
    db_path: Path = DEFAULT_REGISTRY_PATH,
) -> List[Dict[str, Any]]:
    """Find data sources that have columns matching the given roles."""
    if not db_path.exists():
        populate_registry(db_path=db_path)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    qmarks = ",".join("?" for _ in roles)
    query = f"""
        SELECT DISTINCT ds.* 
        FROM data_sources ds
        JOIN source_columns sc ON ds.name = sc.source_name
        WHERE sc.role IN ({qmarks})
        AND ds.is_available = 1
    """
    cur.execute(query, roles)
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]


# ============================================================================
# DATA GENERATION & DOWNLOAD
# ============================================================================

def generate_us_states_data() -> pd.DataFrame:
    """Generate US states reference data."""
    states = [
        ("Alabama", "AL", "01", "South"),
        ("Alaska", "AK", "02", "West"),
        ("Arizona", "AZ", "04", "West"),
        ("Arkansas", "AR", "05", "South"),
        ("California", "CA", "06", "West"),
        ("Colorado", "CO", "08", "West"),
        ("Connecticut", "CT", "09", "Northeast"),
        ("Delaware", "DE", "10", "South"),
        ("Florida", "FL", "12", "South"),
        ("Georgia", "GA", "13", "South"),
        ("Hawaii", "HI", "15", "West"),
        ("Idaho", "ID", "16", "West"),
        ("Illinois", "IL", "17", "Midwest"),
        ("Indiana", "IN", "18", "Midwest"),
        ("Iowa", "IA", "19", "Midwest"),
        ("Kansas", "KS", "20", "Midwest"),
        ("Kentucky", "KY", "21", "South"),
        ("Louisiana", "LA", "22", "South"),
        ("Maine", "ME", "23", "Northeast"),
        ("Maryland", "MD", "24", "South"),
        ("Massachusetts", "MA", "25", "Northeast"),
        ("Michigan", "MI", "26", "Midwest"),
        ("Minnesota", "MN", "27", "Midwest"),
        ("Mississippi", "MS", "28", "South"),
        ("Missouri", "MO", "29", "Midwest"),
        ("Montana", "MT", "30", "West"),
        ("Nebraska", "NE", "31", "Midwest"),
        ("Nevada", "NV", "32", "West"),
        ("New Hampshire", "NH", "33", "Northeast"),
        ("New Jersey", "NJ", "34", "Northeast"),
        ("New Mexico", "NM", "35", "West"),
        ("New York", "NY", "36", "Northeast"),
        ("North Carolina", "NC", "37", "South"),
        ("North Dakota", "ND", "38", "Midwest"),
        ("Ohio", "OH", "39", "Midwest"),
        ("Oklahoma", "OK", "40", "South"),
        ("Oregon", "OR", "41", "West"),
        ("Pennsylvania", "PA", "42", "Northeast"),
        ("Rhode Island", "RI", "44", "Northeast"),
        ("South Carolina", "SC", "45", "South"),
        ("South Dakota", "SD", "46", "Midwest"),
        ("Tennessee", "TN", "47", "South"),
        ("Texas", "TX", "48", "South"),
        ("Utah", "UT", "49", "West"),
        ("Vermont", "VT", "50", "Northeast"),
        ("Virginia", "VA", "51", "South"),
        ("Washington", "WA", "53", "West"),
        ("West Virginia", "WV", "54", "South"),
        ("Wisconsin", "WI", "55", "Midwest"),
        ("Wyoming", "WY", "56", "West"),
        ("District of Columbia", "DC", "11", "South"),
    ]
    return pd.DataFrame(states, columns=["state_name", "state_abbr", "state_fips", "region"])


def generate_sample_weather_data() -> pd.DataFrame:
    """Generate sample weather data for demo purposes."""
    import random
    from datetime import datetime, timedelta
    
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    start_date = datetime(2023, 1, 1)
    
    data = []
    for i in range(365):
        date = start_date + timedelta(days=i)
        for city in cities:
            # Simulate seasonal temperature variation
            base_temp = 50 + 30 * (1 - abs(i - 182) / 182)  # Peak in summer
            temp = base_temp + random.uniform(-10, 10)
            precip = random.uniform(0, 0.5) if random.random() > 0.7 else 0
            conditions = random.choice(["Sunny", "Cloudy", "Rainy", "Partly Cloudy"])
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "city": city,
                "temperature": round(temp, 1),
                "precipitation": round(precip, 2),
                "conditions": conditions,
            })
    
    return pd.DataFrame(data)


def generate_census_income_data() -> pd.DataFrame:
    """Generate sample census income data for demo purposes."""
    import random
    
    # Sample ZIP codes with realistic income distributions
    data = []
    states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    
    for state in states:
        for i in range(50):  # 50 ZIPs per state
            zip_code = f"{random.randint(10000, 99999)}"
            # Income varies by "region"
            base_income = random.randint(35000, 120000)
            population = random.randint(5000, 100000)
            data.append({
                "zip_code": zip_code,
                "median_income": base_income,
                "population": population,
                "state": state,
            })
    
    return pd.DataFrame(data)


def generate_zip_to_fips_data() -> pd.DataFrame:
    """Generate sample ZIP to FIPS mapping data."""
    import random
    
    # Sample data - in production this would be real Census data
    states_fips = {
        "CA": "06", "NY": "36", "TX": "48", "FL": "12", "IL": "17",
        "PA": "42", "OH": "39", "GA": "13", "NC": "37", "MI": "26",
    }
    
    counties = {
        "CA": [("Los Angeles", "037"), ("San Diego", "073"), ("Orange", "059")],
        "NY": [("New York", "061"), ("Kings", "047"), ("Queens", "081")],
        "TX": [("Harris", "201"), ("Dallas", "113"), ("Tarrant", "439")],
        "FL": [("Miami-Dade", "086"), ("Broward", "011"), ("Palm Beach", "099")],
        "IL": [("Cook", "031"), ("DuPage", "043"), ("Lake", "097")],
        "PA": [("Philadelphia", "101"), ("Allegheny", "003"), ("Montgomery", "091")],
        "OH": [("Cuyahoga", "035"), ("Franklin", "049"), ("Hamilton", "061")],
        "GA": [("Fulton", "121"), ("Gwinnett", "135"), ("DeKalb", "089")],
        "NC": [("Mecklenburg", "119"), ("Wake", "183"), ("Guilford", "081")],
        "MI": [("Wayne", "163"), ("Oakland", "125"), ("Macomb", "099")],
    }
    
    data = []
    for state, state_fips in states_fips.items():
        for county_name, county_fips in counties.get(state, []):
            # Generate some ZIP codes for each county
            for _ in range(20):
                zip_code = f"{random.randint(10000, 99999)}"
                fips = state_fips + county_fips
                data.append({
                    "zip": zip_code,
                    "fips": fips,
                    "state": state,
                    "county": county_name,
                })
    
    return pd.DataFrame(data)


def ensure_dataset_exists(source: PublicDataSource) -> bool:
    """Ensure a dataset file exists, generating or downloading if needed."""
    local_path = DATASETS_DIR / source.local_filename
    
    if local_path.exists():
        return True
    
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle generated datasets
    if source.source_url == "generate":
        if source.local_filename == "us_states.csv":
            df = generate_us_states_data()
            df.to_csv(local_path, index=False)
            logger.info(f"Generated {source.local_filename}")
            return True
        if source.local_filename == "zip_to_fips.csv":
            df = generate_zip_to_fips_data()
            df.to_csv(local_path, index=False)
            logger.info(f"Generated {source.local_filename}")
            return True
    
    # Handle bundled datasets (should already exist)
    if source.source_url == "bundled":
        logger.warning(f"Bundled dataset {source.local_filename} not found")
        return False
    
    # Try to download
    if source.source_url.startswith("http"):
        try:
            logger.info(f"Downloading {source.name} from {source.source_url}")
            urlretrieve(source.source_url, local_path)
            logger.info(f"Downloaded {source.local_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {source.name}: {e}")
            return False
    
    return False


def setup_all_datasets() -> Dict[str, bool]:
    """Ensure all registered datasets are available."""
    results = {}
    for source in PUBLIC_DATA_SOURCES:
        results[source.name] = ensure_dataset_exists(source)
    return results


def rebuild_semantic_index() -> None:
    """Rebuild the semantic index with all available datasets."""
    from catalog.semantic_index import build_index
    
    # First ensure datasets exist
    setup_all_datasets()
    
    # Then rebuild the index
    build_index(datasets_dir=str(DATASETS_DIR))
    logger.info("Rebuilt semantic index")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def print_registry_summary():
    """Print a summary of all registered data sources."""
    sources = list_sources()
    
    print("\n" + "=" * 70)
    print("PUBLIC DATA SOURCES REGISTRY")
    print("=" * 70)
    
    by_category = {}
    for src in sources:
        cat = src["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(src)
    
    for category, items in sorted(by_category.items()):
        print(f"\n📁 {category.upper()}")
        print("-" * 40)
        for item in items:
            status = "✅" if item["is_available"] else "❌"
            print(f"  {status} {item['name']}")
            print(f"     └─ {item['description'][:50]}...")
    
    available = sum(1 for s in sources if s["is_available"])
    print(f"\n📊 Total: {len(sources)} sources, {available} available locally")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "list":
            print_registry_summary()
        elif cmd == "setup":
            print("Setting up all datasets...")
            results = setup_all_datasets()
            for name, success in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {name}")
        elif cmd == "rebuild":
            print("Rebuilding semantic index...")
            rebuild_semantic_index()
            print("Done!")
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python -m catalog.public_data_sources [list|setup|rebuild]")
    else:
        print_registry_summary()
