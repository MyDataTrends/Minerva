"""Connectors submodule."""
from public_data.connectors.base import DataConnector, DataSeries, EnrichmentSuggestion

# Cloud storage connectors
try:
    from public_data.connectors.onedrive import OneDriveConnector
except ImportError:
    OneDriveConnector = None

try:
    from public_data.connectors.google_sheets import GoogleSheetsConnector
except ImportError:
    GoogleSheetsConnector = None

# Public API connectors
try:
    from public_data.connectors.fred import FREDConnector
except ImportError:
    FREDConnector = None

try:
    from public_data.connectors.world_bank import WorldBankConnector
except ImportError:
    WorldBankConnector = None
