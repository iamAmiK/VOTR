from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mcp_router.config import RouterConfig, load_config, openai_api_key
from mcp_router.registry.mcp_discovery import (
    DiscoveryError,
    discover_tools_via_sse,
    discover_tools_via_stdio,
    tools_list_result_to_registered_server,
)
from mcp_router.registry.manager import IndexRegistry
from mcp_router.registry.schema import RegisteredServer
from mcp_router.retrieval.embedder import OpenAIEmbedder
from mcp_router.retrieval.engine import RouteResponse, RouterEngine
from mcp_router.retrieval.hybrid import HybridRetriever
from mcp_router.session.memory import SessionMemory


class RouteRequest(BaseModel):
    server_intent: str = Field(..., description="What MCP server domain/capability is needed")
    tool_intent: str = Field(..., description="What operation/tool is needed")
    session_id: Optional[str] = None
    record_session: bool = True


class RegisterRequest(BaseModel):
    server: RegisteredServer


class RegisterDiscoverRequest(BaseModel):
    command: str = Field(..., description="Executable that starts MCP server in stdio mode")
    args: list[str] = Field(default_factory=list)
    server_name: str
    server_description: str
    timeout_seconds: float = Field(default=20.0, ge=2.0, le=120.0)


class RegisterDiscoverSSERequest(BaseModel):
    url: str = Field(..., description="HTTP endpoint for MCP JSON-RPC")
    server_name: str
    server_description: str
    timeout_seconds: float = Field(default=20.0, ge=2.0, le=120.0)


class AppState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cfg: RouterConfig
    engine: RouterEngine
    registry: IndexRegistry


state: Optional[AppState] = None


def create_app(cfg: Optional[RouterConfig] = None) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global state
        if not openai_api_key(cfg):
            raise RuntimeError(
                f"Set {cfg.openai_api_key_env} before starting the router "
                "(required for query embeddings at /route)."
            )
        registry = IndexRegistry(cfg)
        index = registry.load_index()
        embedder = OpenAIEmbedder(cfg)
        sessions = SessionMemory(cfg.session_ttl_seconds)
        engine = RouterEngine(cfg, index, embedder, sessions)
        state = AppState(cfg=cfg, engine=engine, registry=registry)
        yield

    app = FastAPI(title="MCP-Router", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "index_dir": str(cfg.index_dir)}

    @app.post("/route", response_model=RouteResponse)
    def route(req: RouteRequest) -> RouteResponse:
        if state is None:
            raise HTTPException(503, "Router not initialized")
        return state.engine.route(
            server_intent=req.server_intent,
            tool_intent=req.tool_intent,
            session_id=req.session_id,
            record_session=req.record_session,
        )

    @app.post("/register")
    def register(req: RegisterRequest) -> dict[str, Any]:
        if state is None:
            raise HTTPException(503, "Router not initialized")
        embedder = OpenAIEmbedder(state.cfg)
        new_index = state.registry.register_server(req.server, embedder)
        state.engine.index = new_index
        state.engine.hybrid = HybridRetriever(new_index.servers, state.cfg)

        return {"status": "ok", "servers": len(new_index.servers)}

    @app.post("/register/discover")
    def register_discover(req: RegisterDiscoverRequest) -> dict[str, Any]:
        if state is None:
            raise HTTPException(503, "Router not initialized")
        try:
            tools = discover_tools_via_stdio(
                command=req.command,
                args=req.args,
                timeout_seconds=req.timeout_seconds,
            )
            server = tools_list_result_to_registered_server(
                server_name=req.server_name,
                server_description=req.server_description,
                tools_list=tools,
            )
            embedder = OpenAIEmbedder(state.cfg)
            new_index = state.registry.register_server(server, embedder)
            state.engine.index = new_index
            state.engine.hybrid = HybridRetriever(new_index.servers, state.cfg)
            return {
                "status": "ok",
                "servers": len(new_index.servers),
                "discovered_tools": len(server.tools),
                "server_name": server.name,
            }
        except DiscoveryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/register/discover/sse")
    def register_discover_sse(req: RegisterDiscoverSSERequest) -> dict[str, Any]:
        if state is None:
            raise HTTPException(503, "Router not initialized")
        try:
            tools = discover_tools_via_sse(
                url=req.url,
                timeout_seconds=req.timeout_seconds,
            )
            server = tools_list_result_to_registered_server(
                server_name=req.server_name,
                server_description=req.server_description,
                tools_list=tools,
            )
            embedder = OpenAIEmbedder(state.cfg)
            new_index = state.registry.register_server(server, embedder)
            state.engine.index = new_index
            state.engine.hybrid = HybridRetriever(new_index.servers, state.cfg)
            return {
                "status": "ok",
                "servers": len(new_index.servers),
                "discovered_tools": len(server.tools),
                "server_name": server.name,
            }
        except DiscoveryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/session/clear")
    def session_clear(session_id: str) -> dict[str, str]:
        if state is None:
            raise HTTPException(503, "Router not initialized")
        state.engine.sessions.clear_session(session_id)
        return {"status": "ok"}

    return app


app = create_app()
