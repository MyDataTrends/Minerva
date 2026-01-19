"""
Minerva MCP Server - Model Context Protocol implementation.

This module provides an MCP server that exposes Minerva's data analysis,
semantic merge, decisioning, and visualization capabilities as structured
tools for LLM-driven orchestration.

Supports both stdio and HTTP transports.
"""

from .server import MCPServer, create_server
from .session import MCPSession, SessionManager
from .config import MCPConfig

__all__ = [
    "MCPServer",
    "create_server",
    "MCPSession",
    "SessionManager",
    "MCPConfig",
]

__version__ = "0.1.0"
