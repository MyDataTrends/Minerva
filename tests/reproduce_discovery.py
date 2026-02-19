import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mcp_server.discovery_agent import APIDiscoveryAgent

def test_discovery():
    agent = APIDiscoveryAgent()
    
    # Test 1: Registry Match (Steam - recently added)
    print("\n--- Test 1: Registry Match (Steam) ---")
    query_1 = "Top Steam games by month"
    results_1 = agent.discover_api(query_1)
    
    if results_1:
        print(f"✅ Found {len(results_1)} APIs")
        print(f"Top match: {results_1[0].name} (Source: {results_1[0].source}, Conf: {results_1[0].confidence})")
    else:
        print("❌ No APIs found for Steam")

    # Test 2: LLM Fallback (Spotify - not in registry)
    print("\n--- Test 2: LLM Fallback (Spotify) ---")
    query_2 = "Spotify user playlists"
    results_2 = agent.discover_api(query_2)
    
    if results_2:
        print(f"✅ Found {len(results_2)} APIs")
        print(f"Top match: {results_2[0].name} (Source: {results_2[0].source}, Conf: {results_2[0].confidence})")
        if results_2[0].source == 'llm_knowledge':
            print("✅ Correctly used LLM knowledge")
        else:
            print(f"⚠️ Unexpected source: {results_2[0].source}")
    else:
        print("❌ No APIs found for Spotify")

    # Test 3: Key Resolution (Census)
    print("\n--- Test 3: Key Resolution (Census) ---")
    # Simulate an discovered API object
    from mcp_server.discovery_agent import DiscoveredAPI
    census_api = DiscoveredAPI(name="US Census Bureau", description="Test", base_url="test", source="llm_knowledge")
    resolved_key = agent._resolve_api_key("US Census Bureau")
    
    if resolved_key:
        print(f"✅ Resolved key for 'US Census Bureau': {resolved_key[:4]}...*** (" + ("Found" if resolved_key else "Empty") + ")")
    else:
        print("❌ Failed to resolve key for 'US Census Bureau'")


if __name__ == "__main__":
    test_discovery()
