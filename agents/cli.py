"""
CLI entry point for the Minerva agent infrastructure.

Usage:
    python -m agents run conductor          # Run a specific agent
    python -m agents run all                # Run all enabled agents
    python -m agents run conductor --dry-run  # Force dry-run mode
    python -m agents list                   # Show agent status
"""

from __future__ import annotations

import argparse
import sys
import time

from agents.base import AgentResult, BaseAgent
from agents.config import load_agent_configs, get_agent_config
from utils.logging import get_logger

logger = get_logger(__name__)


# Agent registry — maps names to classes
_AGENT_CLASSES = {}


def _register_agents() -> None:
    """Lazily import and register all agent classes."""
    if _AGENT_CLASSES:
        return

    from agents.conductor import ConductorAgent
    from agents.engineer import EngineerAgent
    from agents.sentinel import SentinelAgent
    from agents.advocate import AdvocateAgent
    from agents.productizer import ProductizerAgent
    from agents.marketing import MarketingAgent
    from agents.support import SupportAgent
    from agents.telemetry import TelemetryAgent
    from agents.presentation import PresentationAgent

    _AGENT_CLASSES.update({
        "conductor": ConductorAgent,
        "engineer": EngineerAgent,
        "sentinel": SentinelAgent,
        "advocate": AdvocateAgent,
        "productizer": ProductizerAgent,
        "marketing": MarketingAgent,
        "support": SupportAgent,
        "telemetry": TelemetryAgent,
        "presentation": PresentationAgent,
    })


def _create_agent(name: str, dry_run_override: bool = False) -> BaseAgent:
    """Instantiate an agent by name with its config."""
    _register_agents()

    if name not in _AGENT_CLASSES:
        raise ValueError(f"Unknown agent: {name}. Available: {list(_AGENT_CLASSES.keys())}")

    config = get_agent_config(name)
    if dry_run_override:
        config.dry_run = True

    return _AGENT_CLASSES[name](config=config)


def cmd_run(args: argparse.Namespace, **kwargs) -> None:
    """Execute one or all agents."""
    _register_agents()

    if args.agent == "all":
        agent_names = [n for n in _AGENT_CLASSES if get_agent_config(n).enabled]
    else:
        agent_names = [args.agent]

    for name in agent_names:
        try:
            agent = _create_agent(name, dry_run_override=args.dry_run)
            if not agent.is_enabled:
                print(f"  ⏭  {name}: disabled (skipping)")
                continue

            mode_label = "DRY-RUN" if agent.is_dry_run else "LIVE"
            print(f"  ▶  {name}: running ({mode_label})...")

            start = time.time()
            result = agent.run(**kwargs)
            elapsed = time.time() - start

            status = "✅" if result.success else "❌"
            print(f"  {status}  {name}: {result.summary} ({elapsed:.1f}s)")

            if result.escalations:
                for esc in result.escalations:
                    emoji = {"urgent": "🔴", "review": "🟡", "fyi": "🟢", "metric": "📊"}.get(
                        esc.priority.value, "•"
                    )
                    print(f"      {emoji} {esc.title}")

            if result.error:
                print(f"      ⚠  Error: {result.error}")

        except Exception as exc:
            print(f"  ❌  {name}: failed with exception: {exc}")
            logger.exception("Agent %s failed", name)


def cmd_list(args: argparse.Namespace) -> None:
    """Show status of all agents."""
    _register_agents()
    configs = load_agent_configs()

    print("\n  Minerva Agent Status")
    print("  " + "─" * 50)

    for name in sorted(set(list(_AGENT_CLASSES.keys()) + list(configs.keys()))):
        config = configs.get(name, get_agent_config(name))
        registered = name in _AGENT_CLASSES

        status = "✅ enabled" if config.enabled else "⏸  disabled"
        mode = "dry-run" if config.dry_run else "live"
        impl = "" if registered else " (not implemented)"

        print(f"  {name:12s}  {status}  [{mode}]  schedule={config.schedule}{impl}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agents",
        description="Minerva Agent Infrastructure CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_parser = subparsers.add_parser("run", help="Run one or all agents")
    run_parser.add_argument("agent", help="Agent name or 'all'")
    run_parser.add_argument("--dry-run", action="store_true", help="Force dry-run mode")
    run_parser.set_defaults(func=cmd_run)

    # list command
    list_parser = subparsers.add_parser("list", help="Show agent status")
    list_parser.set_defaults(func=cmd_list)

    args, unknown = parser.parse_known_args()
    
    # Parse unknown args into kwargs
    kwargs = {}
    for arg in unknown:
        if "=" in arg:
            k, v = arg.split("=", 1)
            # Remove leading dashes if any
            k = k.lstrip("-")
            kwargs[k] = v
            
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # Inject kwargs into func
    args.func(args, **kwargs)


if __name__ == "__main__":
    main()
