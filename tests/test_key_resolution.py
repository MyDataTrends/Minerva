
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mcp_server.discovery_agent import APIDiscoveryAgent, DiscoveredAPI

def test_key_resolution():
    agent = APIDiscoveryAgent()
    
    # Mock environment
    os.environ["CENSUS_API_KEY"] = "test_census_key_123"
    os.environ["OPENWEATHERMAP_API_KEY"] = "test_weather_key_456"
    
    print("\n--- Test Key Resolution ---")
    
    # Case 1: Registry Lookup (Census)
    # Registry has "census" -> CENSUS_API_KEY
    key1 = agent._resolve_api_key("US Census Bureau")
    print(f"1. US Census Bureau -> {key1}")
    assert key1 == "test_census_key_123", "Failed to resolve Census key via registry/fuzzy"

    # Case 2: Heuristic (OpenWeatherMap)
    key2 = agent._resolve_api_key("OpenWeatherMap")
    print(f"2. OpenWeatherMap -> {key2}")
    assert key2 == "test_weather_key_456", "Failed to resolve OpenWeatherMap key via heuristic"

    # Case 3: Heuristic (Specific Census fallback)
    key3 = agent._resolve_api_key("The US Census")
    print(f"3. The US Census -> {key3}")
    assert key3 == "test_census_key_123", "Failed to resolve Census key via fallback"
    
    print("✅ All key resolution tests passed!")

if __name__ == "__main__":
    test_key_resolution()
