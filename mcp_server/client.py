"""
MCP Client - Programmatic access to MCP tools.

Enables the local LLM to call MCP tools as part of a self-orchestrating
analysis pipeline. The client provides a simplified interface for tool
discovery and execution.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable

from .server import MCPServer, create_server
from .session import MCPSession, SessionManager, get_session_manager
from .config import MCPConfig, get_config

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for programmatic access to MCP tools.
    
    This client enables:
    - Local LLM self-orchestration
    - Programmatic tool execution
    - Workflow automation
    """
    
    def __init__(
        self,
        server: Optional[MCPServer] = None,
        session: Optional[MCPSession] = None,
        config: Optional[MCPConfig] = None
    ):
        self.config = config or get_config()
        self.server = server or create_server(config=self.config)
        self.session = session
        self._loop = None
    
    def _get_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def create_session(self, user_id: Optional[str] = None) -> MCPSession:
        """Create a new session."""
        session_manager = get_session_manager()
        self.session = session_manager.create_session(user_id=user_id)
        return self.session
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return self.server.list_tools()
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a specific tool."""
        for tool in self.list_tools():
            if tool["name"] == tool_name:
                return tool
        return None
    
    async def call_tool_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool asynchronously.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
                "_meta": {
                    "session_id": self.session.session_id if self.session else None
                }
            }
        }
        
        response = await self.server.handle_request(request)
        
        if "error" in response:
            raise MCPToolError(
                response["error"]["message"],
                response["error"]["code"]
            )
        
        # Parse the result
        content = response["result"]["content"]
        if content and content[0]["type"] == "text":
            return json.loads(content[0]["text"])
        
        return response["result"]
    
    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool synchronously.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        loop = self._get_loop()
        if loop.is_running():
            # If already in async context, create a task
            future = asyncio.ensure_future(
                self.call_tool_async(tool_name, arguments)
            )
            return future
        else:
            return loop.run_until_complete(
                self.call_tool_async(tool_name, arguments)
            )
    
    # Convenience methods for common operations
    
    def load_csv(self, path: str, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """Load a CSV file."""
        return self.call_tool("connect_file", {"path": path})
    
    def describe(self, dataset_id: str) -> Dict[str, Any]:
        """Describe a dataset."""
        return self.call_tool("describe_dataset", {"dataset_id": dataset_id})
    
    def suggest_viz(self, dataset_id: str) -> Dict[str, Any]:
        """Get visualization suggestions."""
        return self.call_tool("suggest_visualizations", {"dataset_id": dataset_id})
    
    def create_chart(
        self,
        dataset_id: str,
        chart_type: str,
        x: Optional[str] = None,
        y: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a chart."""
        return self.call_tool("create_chart", {
            "dataset_id": dataset_id,
            "chart_type": chart_type,
            "x": x,
            "y": y
        })
    
    def run_query(
        self,
        connection_id: str,
        query: str,
        save_as: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a database query."""
        return self.call_tool("execute_query", {
            "connection_id": connection_id,
            "query": query,
            "save_as": save_as
        })


class MCPToolError(Exception):
    """Error raised when a tool call fails."""
    
    def __init__(self, message: str, code: int = -1):
        super().__init__(message)
        self.code = code


# ============================================================================
# Self-Orchestration Engine
# ============================================================================

class SelfOrchestrator:
    """
    Self-orchestrating analysis engine.
    
    Uses the local LLM to plan and execute multi-step analysis
    pipelines by calling MCP tools autonomously.
    """
    
    def __init__(
        self,
        client: Optional[MCPClient] = None,
        max_iterations: int = 10
    ):
        self.client = client or MCPClient()
        self.max_iterations = max_iterations
        self.history: List[Dict[str, Any]] = []
    
    def _get_tool_descriptions(self) -> str:
        """Format tool descriptions for the LLM."""
        tools = self.client.list_tools()
        lines = []
        for tool in tools:
            lines.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(lines)
    
    def _format_system_prompt(self) -> str:
        """Create system prompt for orchestration."""
        return f"""You are an AI data analyst with access to the following tools:

{self._get_tool_descriptions()}

When given a task, plan and execute the necessary steps using these tools.
Respond with a JSON object containing:
- "thought": Your reasoning about what to do next
- "tool": The tool name to call (or null if done)
- "arguments": The arguments for the tool (or null if done)
- "done": true if the task is complete, false otherwise
- "result": Final result summary if done

Only call one tool at a time. Wait for the result before proceeding."""

    async def orchestrate_async(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a self-orchestrating task.
        
        Args:
            task: Natural language description of the task
            context: Optional context (loaded datasets, etc.)
            
        Returns:
            Final result of the orchestration
        """
        # Ensure session exists
        if not self.client.session:
            self.client.create_session()
        
        # Build initial messages
        messages = [
            {"role": "system", "content": self._format_system_prompt()},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        if context:
            messages.append({
                "role": "user",
                "content": f"Context: {json.dumps(context)}"
            })
        
        self.history = []
        
        for i in range(self.max_iterations):
            # Call LLM for next step
            try:
                llm_result = await self.client.call_tool_async("llm_chat", {
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.3
                })
                
                response_text = llm_result.get("response", "")
                
                # Parse LLM response
                try:
                    action = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from the response
                    import re
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if match:
                        action = json.loads(match.group())
                    else:
                        action = {"done": True, "result": response_text}
                
                self.history.append({
                    "iteration": i,
                    "thought": action.get("thought"),
                    "tool": action.get("tool"),
                    "arguments": action.get("arguments")
                })
                
                # Check if done
                if action.get("done"):
                    return {
                        "success": True,
                        "result": action.get("result"),
                        "iterations": i + 1,
                        "history": self.history
                    }
                
                # Execute the tool
                tool_name = action.get("tool")
                tool_args = action.get("arguments", {})
                
                if tool_name:
                    try:
                        tool_result = await self.client.call_tool_async(tool_name, tool_args)
                        
                        # Add result to messages
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({
                            "role": "user",
                            "content": f"Tool result: {json.dumps(tool_result)}"
                        })
                        
                        self.history[-1]["result"] = tool_result
                        
                    except Exception as e:
                        messages.append({
                            "role": "user",
                            "content": f"Tool error: {str(e)}"
                        })
                        self.history[-1]["error"] = str(e)
                
            except Exception as e:
                logger.error(f"Orchestration error at iteration {i}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "iterations": i + 1,
                    "history": self.history
                }
        
        return {
            "success": False,
            "error": "Max iterations reached",
            "iterations": self.max_iterations,
            "history": self.history
        }
    
    def orchestrate(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for orchestrate_async."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.orchestrate_async(task, context))
        finally:
            loop.close()


# ============================================================================
# Convenience Functions
# ============================================================================

def get_client(session_id: Optional[str] = None) -> MCPClient:
    """Get a preconfigured MCP client."""
    client = MCPClient()
    if session_id:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)
        if session:
            client.session = session
    else:
        client.create_session()
    return client


def quick_analyze(file_path: str, question: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick analysis of a data file.
    
    Args:
        file_path: Path to data file (CSV, Excel, etc.)
        question: Optional question to answer about the data
        
    Returns:
        Analysis results
    """
    client = get_client()
    
    # Load the file
    load_result = client.call_tool("connect_file", {"path": file_path})
    if not load_result.get("success"):
        return load_result
    
    dataset_id = load_result["data"]["connection_id"]
    
    # Get description
    desc = client.call_tool("describe_dataset", {"dataset_id": dataset_id})
    
    # Get viz suggestions
    viz = client.call_tool("suggest_visualizations", {"dataset_id": dataset_id})
    
    results = {
        "dataset_id": dataset_id,
        "description": desc.get("data", {}),
        "suggested_visualizations": viz.get("data", {}).get("suggestions", [])
    }
    
    # If there's a question, try to answer it
    if question:
        # Use orchestration for complex questions
        orchestrator = SelfOrchestrator(client=client)
        results["answer"] = orchestrator.orchestrate(question, {"dataset_id": dataset_id})
    
    return results
