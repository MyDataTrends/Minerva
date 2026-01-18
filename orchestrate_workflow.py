"""Backward-compatible shim re-exporting orchestrate_workflow.

This module allows imports like 'from orchestrate_workflow import orchestrate_workflow'
to continue working after the function moved to orchestration.orchestrate_workflow.
"""
import warnings

warnings.warn(
    "Importing from 'orchestrate_workflow' is deprecated. "
    "Use 'from orchestration.orchestrate_workflow import orchestrate_workflow' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from orchestration.orchestrate_workflow import (
    orchestrate_workflow,
    run_workflow,
    setup,
)

__all__ = ["orchestrate_workflow", "run_workflow", "setup"]
