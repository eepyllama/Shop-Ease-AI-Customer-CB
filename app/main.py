"""
FastAPI server — pluggable customer support chatbot backend.
Replaces ComplaintBot's Gradio interface with a REST + WebSocket API
that any company website can embed via widget.html.
"""
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
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
    action: Optional[str] = None                      # "accept" | "select" (from frontend buttons)
    selected_resolution_type: Optional[str] = None    # resolution type chosen via option picker


class ChatResponse(BaseModel):
    session_id: str
    response: str
    decision: Optional[str] = None       # resolution_type (renamed for API compat)
    order_id: Optional[str] = None
    risk_score: Optional[int] = None
    severity: Optional[str] = None
    escalated: bool = False
    pending_offer: Optional[dict] = None  # non-null when awaiting customer confirmation
    resolution_options: Optional[list] = None  # choice list for frontend multi-option UI
    evidence_required: bool = False       # True when upload is expected before options
    timestamp: str


class EvidenceUploadResponse(BaseModel):
    session_id: str
    verdict: str
    confidence: float
    observations: list[str]
    duplicate_image: bool
    exif_flag: bool
    thumbnail_path: str
    timestamp: str


class SessionInfo(BaseModel):
    session_id: str
    phase: Optional[str]
    order_id: Optional[str]


# ── Core chat handler ─────────────────────────────────────────────────────────

async def _run_chat(
    session_id: str,
    user_message: str,
    action: Optional[str] = None,
    selected_resolution_type: Optional[str] = None,
) -> dict:
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
        # Order / customer
        "order_id": session.get("order_id"),
        "customer_id": session.get("customer_id"),
        "customer_name": session.get("customer_name"),
        "customer_email": session.get("customer_email"),
        # Complaint
        "complaint_type": session.get("complaint_type"),
        "claim_description": session.get("claim_description"),
        "order_details": session.get("order_details"),
        # Intake context
        "dietary_restriction": session.get("dietary_restriction"),
        "allergens_declared": session.get("allergens_declared", []),
        "customer_requested_resolution": session.get("customer_requested_resolution"),
        # Risk
        "risk_score": session.get("risk_score", 0),
        "risk_flags": session.get("risk_flags", []),
        # Severity
        "severity": session.get("severity"),
        "allergen_safety": session.get("allergen_safety", False),
        # Escalation
        "escalated": session.get("escalated", False),
        "resolution_attempts": session.get("resolution_attempts", 0),
        # Offer loop
        "pending_offer": session.get("pending_offer"),
        "offer_history": session.get("offer_history", []),
        "confirm_response": None,
        # Button actions (cleared after each turn)
        "action": action,
        "selected_resolution_type": selected_resolution_type,
        # Unresolved issues carried across corrections
        "open_issues": session.get("open_issues", []),
        # Offer-loop dissatisfaction counter
        "dissatisfaction_count": session.get("dissatisfaction_count", 0),
        # Photo-evidence
        "evidence_required": session.get("evidence_required", False),
        "evidence_result":   session.get("evidence_result"),
        "evidence_path":     session.get("evidence_path"),
        # Resolution output
        "resolution_type": None,
        "compensation_amount": None,
        "reasoning": None,
        # RAG
        "rag_context": None,
        "policy_confidence": 0.0,
        # Response
        "final_response": None,
        "token_log": [],
    }

    # Run graph
    result = await graph.ainvoke(state)

    # Persist updated state
    resolved_types = {"full_refund", "partial_refund", "coupon", "store_credit"}
    resolution_type = result.get("resolution_type")
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
        "dietary_restriction": result.get("dietary_restriction") or session.get("dietary_restriction"),
        "allergens_declared": result.get("allergens_declared") or session.get("allergens_declared", []),
        "customer_requested_resolution": result.get("customer_requested_resolution") or session.get("customer_requested_resolution"),
        "risk_score": result.get("risk_score", 0),
        "risk_flags": result.get("risk_flags", []),
        "severity": result.get("severity") or session.get("severity"),
        "allergen_safety": result.get("allergen_safety", session.get("allergen_safety", False)),
        "escalated": result.get("escalated", session.get("escalated", False)),
        "resolution_attempts": result.get("resolution_attempts", session.get("resolution_attempts", 0)),
        "pending_offer": result.get("pending_offer"),
        "offer_history": result.get("offer_history", session.get("offer_history", [])),
        "open_issues": result.get("open_issues", session.get("open_issues", [])),
        "dissatisfaction_count": result.get("dissatisfaction_count", session.get("dissatisfaction_count", 0)),
        "evidence_required": result.get("evidence_required", session.get("evidence_required", False)),
        "evidence_result":   result.get("evidence_result") or session.get("evidence_result"),
        "evidence_path":     result.get("evidence_path") or session.get("evidence_path"),
        # Never persist button-action fields — they are single-turn only
        "action": None,
        "selected_resolution_type": None,
        "phase": "resolved" if resolution_type in resolved_types else
                 "escalated" if result.get("escalated") else "active",
    }
    save_session(session_id, updated_session)

    pending_offer = result.get("pending_offer")
    evidence_required = bool(
        pending_offer and pending_offer.get("evidence_required")
    )
    return {
        "response": result.get("final_response", "I'm sorry, something went wrong. Please try again."),
        "decision": resolution_type,
        "order_id": result.get("order_id") or session.get("order_id"),
        "risk_score": result.get("risk_score"),
        "severity": result.get("severity"),
        "escalated": result.get("escalated", False),
        "pending_offer": pending_offer,
        "resolution_options": pending_offer.get("resolution_options") if pending_offer else None,
        "evidence_required": evidence_required,
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
        result = await _run_chat(
            session_id, req.message,
            action=req.action,
            selected_resolution_type=req.selected_resolution_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        decision=result.get("decision"),
        order_id=result.get("order_id"),
        risk_score=result.get("risk_score"),
        severity=result.get("severity"),
        escalated=result.get("escalated", False),
        pending_offer=result.get("pending_offer"),
        resolution_options=result.get("resolution_options"),
        evidence_required=result.get("evidence_required", False),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/api/session/{session_id}/evidence", response_model=EvidenceUploadResponse)
async def upload_evidence(
    session_id: str,
    file: UploadFile = File(...),
):
    """
    Accept a photo upload for evidence verification.
    Validates format/size, runs perceptual-hash, EXIF and vision-LLM checks,
    stores the result in the session, and returns the verdict.
    The frontend should follow up with action='evidence_uploaded' (or 'skip_evidence')
    via POST /api/chat to trigger the next graph turn.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    order_id = session.get("order_id") or "unknown"
    order_details = session.get("order_details") or {}

    image_bytes = await file.read()
    content_type = file.content_type or "application/octet-stream"

    from app.evidence import validate_and_store, check_evidence

    try:
        stored = validate_and_store(session_id, order_id, image_bytes, content_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Determine order date for EXIF check
    order_date = order_details.get("created_at") or order_details.get("order_date")

    ev_result = check_evidence(
        image_path=stored["image_path"],
        order_details=order_details,
        claim_description=session.get("claim_description", ""),
        order_date=order_date,
    )

    # Persist evidence result in session so the next graph turn can read it
    session["evidence_result"] = ev_result.to_dict()
    session["evidence_path"]   = stored["thumbnail_path"]
    save_session(session_id, session)

    return EvidenceUploadResponse(
        session_id=session_id,
        verdict=ev_result.verdict,
        confidence=ev_result.confidence,
        observations=ev_result.observations,
        duplicate_image=ev_result.duplicate_image,
        exif_flag=ev_result.exif_flag,
        thumbnail_path=stored["thumbnail_path"],
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
                    "severity": result.get("severity"),
                    "escalated": result.get("escalated", False),
                    "pending_offer": result.get("pending_offer"),
                    "resolution_options": result.get("resolution_options"),
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
