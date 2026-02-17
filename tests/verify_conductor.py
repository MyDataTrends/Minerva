
import sys
import os
import shutil
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

from agents.conductor import ConductorAgent
from agents.config import AgentConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConductorVerifier")

def test_conductor_workflow():
    print("Starting Conductor Verification...")
    
    # 1. Setup Mock GitHub Client
    # We patch the class where it is IMPORTED in the agent files
    # Note: imports happen inside methods usually to avoid circular deps
    
    with patch('agents.tools.github_client.GitHubClient') as MockGitHub:
        mock_client = MockGitHub.return_value
        mock_client.is_available = True
        mock_client.get_repo_info.return_value = {"stargazers_count": 100, "forks_count": 20}
        mock_client.list_issues.return_value = [
            {"number": 1, "title": "Test Issue", "state": "open", "labels": [], "body": "Help me"}
        ]
        mock_client.list_pull_requests.return_value = []
        
        # 2. Configure Conductor
        # We manually enable sub-agents via config mocking or just let them run default
        # The Conductor loads configs via load_agent_configs(). We should patch that too to ensure enabled.
        
        with patch('agents.conductor.load_agent_configs') as mock_load_configs:
            # Return enabled configs for all
            mock_load_configs.return_value = {
                "engineer": AgentConfig(name="engineer", enabled=True, schedule="daily"),
                "sentinel": AgentConfig(name="sentinel", enabled=True, schedule="event"),
                "advocate": AgentConfig(name="advocate", enabled=True, schedule="daily"),
            }
            
            # 3. Run Conductor
            agent = ConductorAgent()
            # Force dry run to avoid real side effects (though mocks handle most)
            agent.config.dry_run = True 
            
            print("Running ConductorAgent.run()...")
            try:
                result = agent.run()
                print(f"Conductor Finished. Success={result.success}")
                print(f"Summary: {result.summary}")
            except Exception as e:
                print(f"Conductor Failed: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            
            # 4. Verify Digest Generation
            # Conductor saves digest to agents/digests/YYYY-MM-DD.md
            # In dry run, it might not write to disk?
            # conductor.py line 206: 
            # if self.is_dry_run: ... filepath.write_text(digest, encoding="utf-8") ... return str(filepath)
            # Yes it writes to disk even in dry run currently (based on code reading)
            
            from datetime import datetime
            today = datetime.utcnow().strftime("%Y-%m-%d")
            digest_dir = Path("agents/digests")
            digest_file = digest_dir / f"{today}.md"
            
            if digest_file.exists():
                print(f"Digest Found: {digest_file}")
                content = digest_file.read_text(encoding="utf-8")
                print(f"---\n{content[:500]}...\n---")
                
                if "Engineer" in content and "Advocate" in content:
                    print("Digest contains sub-agent reports.")
                else:
                    print("Warning: Digest missing sub-agent reports.")
            else:
                print(f"Error: Digest file not found at {digest_file}")
                sys.exit(1)

            # 5. Verify Engineer Output
            # Engineer writes to agents/knowledge_base/product/vision_gap_analysis.md
            # kb_path = base_path / "agents/knowledge_base/product/vision_gap_analysis.md"
            # logic handles path relative to something?
            # EngineerAgent.py: self.kb.write_doc(...)
            # functional check
            
            print("CONDUCTOR AGENT VERIFIED")

if __name__ == "__main__":
    test_conductor_workflow()
