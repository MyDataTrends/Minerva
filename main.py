import argparse
import os


def _apply_cli_env_flags() -> None:
    """Parse CLI flags and map them to environment variables."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-llm", action="store_true", help="Disable local LLM usage")
    parser.add_argument(
        "--enable-prometheus",
        action="store_true",
        help="Start Prometheus metrics server",
    )
    parser.add_argument(
        "--safe-logs", action="store_true", help="Enable redaction in log output"
    )
    parser.add_argument(
        "--dev-lenient",
        action="store_true",
        help="Relax checks for local development",
    )
    parser.add_argument(
        "--mcp-server",
        action="store_true",
        help="Start MCP server for AI tool access",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8766,
        help="MCP server HTTP port (default: 8766)",
    )
    args, _ = parser.parse_known_args()

    if args.no_llm:
        os.environ["ENABLE_LOCAL_LLM"] = "0"
    else:
        os.environ.setdefault("ENABLE_LOCAL_LLM", "1")

    if args.enable_prometheus:
        os.environ["ENABLE_PROMETHEUS"] = "1"
    else:
        os.environ.setdefault("ENABLE_PROMETHEUS", "0")

    if args.safe_logs:
        os.environ["REDACTION_ENABLED"] = "1"
    if args.dev_lenient:
        os.environ["LOCAL_DEV_LENIENT"] = "1"
    
    # MCP server settings
    if args.mcp_server:
        os.environ["MCP_ENABLED"] = "1"
        os.environ["MCP_PORT"] = str(args.mcp_port)


_apply_cli_env_flags()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt, StrictStr, constr
from utils.logging import configure_logging
from utils.metrics import REQUESTS
from config import ENABLE_PROMETHEUS, get_bool
from prometheus_client import start_http_server
import json

from storage import session_db

configure_logging()

_metrics_started = False
_mcp_server = None
_mcp_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global _metrics_started, _mcp_server, _mcp_task
    import asyncio
    
    # Startup
    if ENABLE_PROMETHEUS and not _metrics_started:
        addr = "0.0.0.0" if get_bool("METRICS_PUBLIC", False) else "127.0.0.1"
        start_http_server(8000, addr=addr)
        _metrics_started = True
    
    # Start MCP server if enabled
    if get_bool("MCP_ENABLED", False):
        try:
            from mcp_server import create_server
            from mcp_server.config import MCPConfig
            
            config = MCPConfig()
            _mcp_server = create_server(config=config)
            _mcp_task = asyncio.create_task(_mcp_server.run_http())
            import logging
            logging.getLogger(__name__).info(f"MCP server started on http://{config.host}:{config.port}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to start MCP server: {e}")
    
    yield
    
    # Shutdown
    if _mcp_server:
        _mcp_server.stop()
        if _mcp_task:
            _mcp_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_: Request, __: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "Invalid payload"})


Scalar = StrictBool | StrictInt | StrictFloat | StrictStr


class RerunPayload(BaseModel):
    """Payload for rerunning a workflow session."""
    model_config = ConfigDict(extra="forbid", strict=True)
    
    user_id: constr(strict=True, max_length=100) | None = None
    file_name: constr(strict=True, max_length=100) | None = None
    target_column: constr(strict=True, max_length=100) | None = None
    category: constr(strict=True, max_length=100) | None = None
    user_labels: dict[constr(strict=True, max_length=100), constr(strict=True, max_length=100)] | None = None
    diagnostics_config: dict[constr(strict=True, max_length=100), Scalar] | None = None


@app.get("/healthz")
def healthz():
    REQUESTS.inc()
    return {"status": "ok"}


@app.get("/sessions")
def list_sessions(limit: int = 20):
    """Return recent workflow sessions."""
    return session_db.list_sessions(limit)


@app.get("/sessions/{run_id}")
def get_session(run_id: str):
    session = session_db.get_session(run_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/sessions/{run_id}/rerun")
def rerun_session(run_id: str, payload: RerunPayload | None = None):
    session = session_db.get_session(run_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    from orchestration.orchestrate_workflow import orchestrate_workflow
    from storage.local_backend import load_datalake_dfs

    params = json.loads(session.get("params") or "{}")
    if payload:
        params.update(payload.model_dump(exclude_none=True))
    result = orchestrate_workflow(datalake_dfs=load_datalake_dfs(), **params)
    session_db.record_session(result.get("run_id"), params, result)
    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
