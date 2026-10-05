"""
FastAPI server — pluggable customer support chatbot backend.
Replaces ComplaintBot's Gradio interface with a REST + WebSocket API
that any company website can embed via widget.html.
"""
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.config import ALLOWED_ORIGINS
from app.graph import get_graph, AgentState
from app.rag import build_retriever
from app.tools import get_session, save_session, list_sessions
from app.ops_tools import (
    get_support_policy, get_pending_escalations,
    get_pending_human_checks, get_support_stats,
)


# ── Startup: warm up RAG index ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Building RAG index from policy document…")
    try:
        build_retriever()
        print("✓ RAG index ready.")
    except Exception as e:
        print(f"⚠ RAG index build failed: {e}. Continuing without RAG.")
    yield


app = FastAPI(
    title="ResolveIQ Chat API",
    description="AI-powered customer support chatbot — plug into any website",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    company_name: Optional[str] = "Support"  # for widget customisation


class ChatResponse(BaseModel):
    session_id: str
    response: str
    decision: Optional[str] = None
    order_id: Optional[str] = None
    risk_score: Optional[int] = None
    timestamp: str


class SessionInfo(BaseModel):
    session_id: str
    phase: Optional[str]
    order_id: Optional[str]


# ── Core chat handler ─────────────────────────────────────────────────────────

async def _run_chat(session_id: str, user_message: str) -> dict:
    """Load session state, run the LangGraph pipeline, persist state."""
    graph = get_graph()

    # Load or create session state
    session = get_session(session_id)

    # Append user message to history
    existing_messages = session.get("messages", [])
    existing_messages.append(HumanMessage(content=user_message))

    # Build full state for this turn
    state: AgentState = {
        "session_id": session_id,
        "messages": existing_messages,
        "order_id": session.get("order_id"),
        "customer_id": session.get("customer_id"),
        "customer_name": session.get("customer_name"),
        "customer_email": session.get("customer_email"),
        "complaint_type": session.get("complaint_type"),
        "claim_description": session.get("claim_description"),
        "order_details": session.get("order_details"),
        "risk_score": session.get("risk_score", 0),
        "risk_flags": session.get("risk_flags", []),
        "decision": None,
        "refund_amount": None,
        "reasoning": None,
        "rag_context": None,
        "final_response": None,
        "token_log": [],
    }

    # Run graph
    result = await graph.ainvoke(state)

    # Persist updated state
    updated_session = {
        **session,
        "messages": result.get("messages", existing_messages),
        "order_id": result.get("order_id") or session.get("order_id"),
        "customer_id": result.get("customer_id") or session.get("customer_id"),
        "customer_name": result.get("customer_name") or session.get("customer_name"),
        "customer_email": result.get("customer_email") or session.get("customer_email"),
        "complaint_type": result.get("complaint_type") or session.get("complaint_type"),
        "claim_description": result.get("claim_description") or session.get("claim_description"),
        "order_details": result.get("order_details") or session.get("order_details"),
        "risk_score": result.get("risk_score", 0),
        "risk_flags": result.get("risk_flags", []),
        "phase": "resolved" if result.get("decision") in ("full_refund", "partial_refund", "deny") else "active",
    }
    save_session(session_id, updated_session)

    return {
        "response": result.get("final_response", "I'm sorry, something went wrong. Please try again."),
        "decision": result.get("decision"),
        "order_id": result.get("order_id") or session.get("order_id"),
        "risk_score": result.get("risk_score"),
    }


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Send a customer message and receive an AI response.
    Pass the returned session_id back on every subsequent message.
    """
    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = await _run_chat(session_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        decision=result.get("decision"),
        order_id=result.get("order_id"),
        risk_score=result.get("risk_score"),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Return conversation history for a session."""
    session = get_session(session_id)
    if not session.get("messages"):
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in session["messages"]
    ]
    return {
        "session_id": session_id,
        "order_id": session.get("order_id"),
        "phase": session.get("phase"),
        "messages": messages,
    }


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_all_sessions():
    """Admin endpoint: list all active sessions."""
    return list_sessions()


@app.get("/api/escalations")
async def get_escalations():
    """Ops endpoint: orders escalated to the human review team."""
    return get_pending_escalations()


@app.get("/api/human-checks")
async def get_human_checks():
    """Ops endpoint: all orders and actions requiring human attention."""
    return get_pending_human_checks()


@app.get("/api/stats")
async def support_stats():
    """Ops endpoint: summary metrics — decision counts, compensation totals, open cases."""
    return get_support_stats()


@app.get("/api/policy/{issue_type}")
async def policy(issue_type: str):
    """Ops endpoint: structured policy thresholds for a given issue type."""
    return get_support_policy(issue_type)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
async def root():
    """Serve the chat widget demo at the root URL."""
    from pathlib import Path
    widget_path = Path("widget.html")
    if widget_path.exists():
        return FileResponse("widget.html", media_type="text/html")
    return {"message": "ResolveIQ API is running. Try POST /api/chat or visit /docs"}


@app.get("/widget.js")
async def serve_widget_js():
    """Serve the injectable JS widget for embedding in any website."""
    from pathlib import Path
    js_path = Path("widget.js")
    if js_path.exists():
        return FileResponse("widget.js", media_type="application/javascript")
    return {"error": "widget.js not found"}


# ── WebSocket endpoint (real-time chat) ──────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time streaming chat.
    Client sends: {"message": "..."}
    Server sends: {"response": "...", "decision": "...", "order_id": "..."}
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()
            if not user_message:
                continue

            # Send typing indicator
            await websocket.send_json({"type": "typing", "status": True})

            try:
                result = await _run_chat(session_id, user_message)
                await websocket.send_json({
                    "type": "message",
                    "response": result["response"],
                    "decision": result.get("decision"),
                    "order_id": result.get("order_id"),
                    "risk_score": result.get("risk_score"),
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "response": "I'm having trouble right now. Please try again in a moment.",
                })
            finally:
                await websocket.send_json({"type": "typing", "status": False})

    except WebSocketDisconnect:
        pass


# ── Serve widget ──────────────────────────────────────────────────────────────

@app.get("/widget")
async def serve_widget():
    """Serve the embeddable demo page."""
    from pathlib import Path
    widget_path = Path("widget.html")
    if widget_path.exists():
        return FileResponse("widget.html")
    return {"error": "widget.html not found"}
