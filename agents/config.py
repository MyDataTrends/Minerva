"""
Agent configuration loader.

Reads agent settings from agents/agents_config.yaml and merges
with environment variables. Global overrides (AGENT_DRY_RUN) take
precedence over per-agent settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import yaml

from agents.base import AgentConfig
from utils.logging import get_logger

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "agents_config.yaml"


def _load_yaml() -> dict:
    """Load the YAML config file, return empty dict if missing."""
    if not CONFIG_FILE.exists():
        logger.warning("Agent config file not found at %s, using defaults", CONFIG_FILE)
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to parse agent config: %s", exc)
        return {}


def load_agent_configs() -> Dict[str, AgentConfig]:
    """
    Load all agent configurations.

    Priority order (highest wins):
    1. Environment variables (AGENT_DRY_RUN overrides all dry_run settings)
    2. agents_config.yaml per-agent settings
    3. Defaults in AgentConfig dataclass
    """
    raw = _load_yaml()
    global_dry_run = os.getenv("AGENT_DRY_RUN", "").lower() in ("1", "true", "yes")

    configs: Dict[str, AgentConfig] = {}
    agents_section = raw.get("agents", {})

    for agent_name, agent_raw in agents_section.items():
        if not isinstance(agent_raw, dict):
            continue

        dry_run = global_dry_run or agent_raw.get("dry_run", True)

        configs[agent_name] = AgentConfig(
            name=agent_name,
            enabled=agent_raw.get("enabled", True),
            dry_run=dry_run,
            schedule=agent_raw.get("schedule", "daily"),
            llm_model=agent_raw.get("llm_model", "claude-3-5-sonnet"),
            extra=agent_raw.get("extra", {}),
        )

    return configs


def get_agent_config(agent_name: str) -> AgentConfig:
    """Get config for a specific agent, with defaults if not in YAML."""
    configs = load_agent_configs()
    if agent_name in configs:
        return configs[agent_name]

    # Return defaults with global dry_run check
    global_dry_run = os.getenv("AGENT_DRY_RUN", "").lower() in ("1", "true", "yes")
    return AgentConfig(name=agent_name, dry_run=global_dry_run or True)


def get_github_config() -> Dict[str, Optional[str]]:
    """Get GitHub API configuration from environment."""
    return {
        "token": os.getenv("GITHUB_TOKEN"),
        "repo": os.getenv("GITHUB_REPO", "MyDataTrends/Minerva"),
    }


def get_anthropic_key() -> Optional[str]:
    """Get Anthropic API key from environment."""
    return os.getenv("ANTHROPIC_API_KEY")
