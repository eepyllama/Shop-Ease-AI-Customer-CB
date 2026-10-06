"""
LangGraph pipeline — revised architecture.

Graph entry: escalation_check (runs EVERY turn first)

Full routing:
  escalation_check
    → acknowledge_node       (hard stop — session already escalated or allergen)
    → confirm_node           (pending_offer exists — customer responding to an offer)
    → intake_node            (normal path)

  intake_node
    → request_info_node      (no order_id)
    → adjudication_node      (has order_id)

  adjudication_node
    → escalate_node          (resolution_type == "escalate" OR allergen_safety OR human_required)
    → offer_node             (all other resolution types — store as pending_offer, END)

  confirm_node
    → execute_node           (customer accepted)
    → adjudication_node      (rejected, retry allowed — adjudication picks different option)
    → escalate_node          (rejected, max rejections hit)
    → intake_node            (new complaint or correction — resets complaint state)

  execute_node  → END
  escalate_node → END
  acknowledge_node → END
  request_info_node → END
  offer_node → END
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.config import (
    GROQ_API_KEY, LLM_MODEL, LLM_PROVIDER, OPENAI_API_KEY,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL,
    ESCALATIONS_LOG, MAX_REJECTION_ATTEMPTS, ALLERGEN_ALWAYS_ESCALATE,
    POLICY_CONFIDENCE_MIN, DEV_FAIL_LOUD,
    EVIDENCE_RISK_THRESHOLD,
    LATE_DELIVERY_NONE_MIN, LATE_DELIVERY_LOW_MIN, LATE_DELIVERY_MEDIUM_MIN,
)
from app.verification import (
    VerificationResult, get_verifier_type, verify_timestamps, _PHOTO_TYPES,
    verify_allergen, verify_food,
)
from app.escalation import check_escalation
from app.handoff import generate_handoff_summary, build_escalation_record
from app.prompts import (
    INTAKE_SYSTEM, INTAKE_USER_TEMPLATE,
    ADJUDICATION_SYSTEM, ADJUDICATION_USER_TEMPLATE,
    OFFER_SYSTEM, OFFER_USER_TEMPLATE,
    CONFIRM_SYSTEM, CONFIRM_USER_TEMPLATE,
    EXECUTE_RESPONSE_SYSTEM, EXECUTE_USER_TEMPLATE,
    REQUEST_ORDER_ID_RESPONSE,
)
from app.rag import get_rag_context_with_confidence
from app.resolution import (
    compute_allowed_options, compute_compensation,
    validate_resolution, requires_human_approval,
)
from app.resolution_matrix import (
    get_options, get_options_after_dissatisfaction, compute_option_amount,
)
from app.risk import compute_risk_score
from app.severity import classify_severity, extract_intake_context
from app.tools import (
    get_order_details, get_refund_history,
    set_refund_status, set_partial_refund_status,
    set_humancheck_status, log_escalation,
    send_voucher_email, _conn,
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    # Order / customer
    order_id: Optional[str]
    customer_id: Optional[str]
    customer_name: Optional[str]
    customer_email: Optional[str]
    # Complaint classification
    complaint_type: Optional[str]
    claim_description: Optional[str]
    order_details: Optional[dict]
    # Phase 1: expanded intake fields
    dietary_restriction: Optional[str]
    allergens_declared: list[str]
    customer_requested_resolution: Optional[str]
    # Risk
    risk_score: int
    risk_flags: list[str]
    # Severity
    severity: Optional[str]
    allergen_safety: bool
    # Escalation state
    escalated: bool
    resolution_attempts: int
    # Offer / confirm loop
    pending_offer: Optional[dict]
    offer_history: list[dict]
    confirm_response: Optional[str]
    # Resolution output
    resolution_type: Optional[str]
    compensation_amount: Optional[float]
    reasoning: Optional[str]
    # RAG
    rag_context: Optional[str]
    policy_confidence: float
    # Correction tracking
    correction_detected: bool
    # Multi-issue tracking: unresolved complaints pushed aside by a correction
    open_issues: list[dict]
    # Frontend action override (button click bypasses LLM classification)
    action: Optional[str]                  # "accept" | "select"
    selected_resolution_type: Optional[str]  # set when action == "select"
    # Typed dissatisfaction counter for current complaint (resets on new complaint / execution)
    dissatisfaction_count: int
    # Photo-evidence verification (raw upload result — kept for /api/session/evidence endpoint)
    evidence_required: bool
    evidence_result: Optional[dict]   # EvidenceResult.to_dict() after upload + check
    evidence_path: Optional[str]      # thumbnail path for handoff brief
    # Cross-check verification layer (runs before adjudication)
    verification_result: Optional[dict]   # VerificationResult.to_dict()
    case_status: Optional[str]            # None | "awaiting_evidence"
    # Response
    final_response: Optional[str]
    token_log: list[dict]


# ── Pydantic output schemas ───────────────────────────────────────────────────

class IntakeOutput(BaseModel):
    complaint_type: str = Field(description="Complaint category")
    order_id: Optional[str] = Field(None)
    claim_description: str = Field(description="≤30-word summary")
    confidence: float = Field(ge=0.0, le=1.0)
    dietary_restriction: Optional[str] = Field(None)
    allergens_declared: list[str] = Field(default_factory=list)
    customer_requested_resolution: Optional[str] = Field(None)
    is_explicit_human_request: bool = Field(default=False)


class AdjudicationOutput(BaseModel):
    resolution_type: str = Field(description="One value from allowed_resolutions list")
    reasoning: str = Field(description="Internal reasoning ≤100 words")
    risk_flags: list[str] = Field(default_factory=list)


class ConfirmOutput(BaseModel):
    response_type: str = Field(description="accept | dissatisfied | new_complaint")
    reason: Optional[str] = Field(None)


# ── LLM factory ──────────────────────────────────────────────────────────────

# Groq HTTP status codes that should trigger OpenRouter fallback
_GROQ_RETRYABLE_CODES: frozenset[int] = frozenset({
    401,   # invalid / expired key
    429,   # rate limit
    500,   # groq internal error
    502,
    503,
    504,
})


def _is_groq_retriable(exc: Exception) -> bool:
    """Return True for Groq errors that should fall over to OpenRouter."""
    msg = str(exc).lower()
    # groq SDK wraps HTTP errors; check status code attr first, then message
    code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if code in _GROQ_RETRYABLE_CODES:
        return True
    return any(kw in msg for kw in (
        "invalid api key", "authentication", "rate limit",
        "quota", "overloaded", "service unavailable",
    ))


def _build_openrouter_llm():
    if not OPENROUTER_API_KEY:
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )


class _FallbackLLM:
    """
    Thin wrapper: calls `primary` first.  On a retriable Groq error it
    transparently falls over to `fallback` (OpenRouter).  If `fallback` is
    None the original exception is re-raised.
    """

    def __init__(self, primary, fallback, structured_output_class=None):
        self._primary  = primary
        self._fallback = fallback
        self._soc      = structured_output_class

    # Support llm.with_structured_output(cls) chaining
    def with_structured_output(self, schema):
        p = self._primary.with_structured_output(schema)
        f = self._fallback.with_structured_output(schema) if self._fallback else None
        return _FallbackLLM(p, f, structured_output_class=schema)

    def invoke(self, *args, **kwargs):
        try:
            return self._primary.invoke(*args, **kwargs)
        except Exception as exc:
            if self._fallback and _is_groq_retriable(exc):
                logger.warning(
                    "Groq call failed (%s) — falling back to OpenRouter.", exc
                )
                return self._fallback.invoke(*args, **kwargs)
            raise

    async def ainvoke(self, *args, **kwargs):
        try:
            return await self._primary.ainvoke(*args, **kwargs)
        except Exception as exc:
            if self._fallback and _is_groq_retriable(exc):
                logger.warning(
                    "Groq ainvoke failed (%s) — falling back to OpenRouter.", exc
                )
                return await self._fallback.ainvoke(*args, **kwargs)
            raise

    # Pass through anything else (stream, batch, …)
    def __getattr__(self, name):
        return getattr(self._primary, name)


def _get_llm(structured_output_class=None):
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        if structured_output_class:
            return llm.with_structured_output(structured_output_class)
        return llm

    # Primary: Groq
    from langchain_groq import ChatGroq
    primary = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)
    fallback = _build_openrouter_llm()

    wrapped = _FallbackLLM(primary, fallback)
    if structured_output_class:
        return wrapped.with_structured_output(structured_output_class)
    return wrapped


# ── Audit log helper ─────────────────────────────────────────────────────────

def _log_audit(session_id: str, order_id: Optional[str], customer_id: Optional[str],
               event_type: str, payload: dict) -> None:
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO audit_log (log_id, session_id, order_id, customer_id, event_type, payload, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    session_id,
                    order_id,
                    customer_id,
                    event_type,
                    json.dumps(payload),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except Exception:
        pass  # audit failures must never break the chat flow


# ── Order context filter ──────────────────────────────────────────────────────
# Maps complaint_type → the order fields shown to the adjudication LLM.
# Prevents unrelated attributes (e.g. delivery timing when the complaint is
# dietary_mismatch) from contaminating the LLM's reasoning.
_COMPLAINT_ORDER_FIELDS: dict[str, set] = {
    "late_delivery":      {"order_id", "status", "estimated_delivery_min",
                           "actual_delivery_min", "promised_eta_at_order_time",
                           "actual_delivered_at", "total"},
    "missing_item":       {"order_id", "status", "items", "total"},
    "quality_issue":      {"order_id", "status", "items", "total"},
    "wrong_order":        {"order_id", "status", "items", "total"},
    "dietary_mismatch":   {"order_id", "status", "items", "total"},
    "allergen_violation": {"order_id", "status", "items", "total"},
    "foreign_object":     {"order_id", "status", "items", "total"},
    "freebie_missing":    {"order_id", "status", "items", "total"},
    "payment_issue":      {"order_id", "status", "total"},
    "refund_status":      {"order_id", "status", "refund_amount", "total"},
}


def _filter_order_context(order: dict, complaint_type: str) -> dict:
    """Return only order fields relevant to the complaint type.

    This is the primary guard against the LLM seeing — and reasoning about —
    order facts unrelated to the customer's actual complaint (e.g., a 95-min
    delivery delay when the complaint is about a dietary mismatch).
    """
    allowed = _COMPLAINT_ORDER_FIELDS.get(complaint_type)
    if allowed is None:
        return order  # "other" — show everything
    return {k: v for k, v in order.items() if k in allowed}


# ── Correction detection ──────────────────────────────────────────────────────
_CORRECTION_RE = re.compile(
    r"\b(its not|it.?s not|it is not"
    r"|not (the|my|a|an) (issue|problem|complaint|case)"
    r"|wrong issue|wrong (problem|complaint)"
    r"|not about (the|a|an)?"
    r"|you (mis|got|have the wrong)"
    r"|never said|didn.?t say|i said|i never"
    r"|actual (issue|problem|complaint)"
    r"|the (real|actual|true) (issue|problem|complaint)"
    r"|misunderstood|not what i said|not what i complained)\b",
    re.IGNORECASE,
)


def _is_correction(text: str) -> bool:
    return bool(_CORRECTION_RE.search(text))


# ── Resolution label helper ───────────────────────────────────────────────────

_RESOLUTION_LABELS = {
    "full_refund":       "Full Refund",
    "partial_refund":    "Partial Refund",
    "coupon":            "Voucher / Coupon",
    "store_credit":      "Store Credit",
    "replacement":       "Replacement Order",
    "priority_handling": "Priority Handling",
    "apology":           "Apology",
    "explanation":       "Explanation",
    "escalate":          "Talk to a Human Agent",
}


def _resolution_label(rt: str) -> str:
    return _RESOLUTION_LABELS.get(rt, rt.replace("_", " ").title())


# Kept for backward compat; verification layer now uses _PHOTO_TYPES from app.verification
_EVIDENCE_REQUIRED_TYPES = _PHOTO_TYPES

# Severity tier ordering (high → low) for "cap one tier lower" rule
_SEVERITY_TIERS = ("HIGH", "MEDIUM", "LOW")


def _lower_severity(s: str) -> str:
    """Return the next lower severity tier, or LOW if already at the bottom."""
    try:
        idx = _SEVERITY_TIERS.index(s)
        return _SEVERITY_TIERS[min(idx + 1, len(_SEVERITY_TIERS) - 1)]
    except ValueError:
        return "LOW"


# ── Node: escalation_check ────────────────────────────────────────────────────

def escalation_check_node(state: AgentState) -> dict:
    result = check_escalation(state)
    return {
        "confirm_response": None,  # reset each turn
        "_escalation_check": result,
    }


def route_after_escalation_check(state: AgentState) -> str:
    result = state.get("_escalation_check", {})
    if result.get("should_escalate") and result.get("is_hard_stop"):
        return "acknowledge"
    # Resume photo verification when awaiting evidence and the user acted
    if state.get("case_status") == "awaiting_evidence" and state.get("action") in (
        "evidence_uploaded", "skip_evidence"
    ):
        return "verification"
    if state.get("pending_offer") and not result.get("should_escalate"):
        return "confirm"
    return "intake"


# ── Node: acknowledge ─────────────────────────────────────────────────────────

def acknowledge_node(state: AgentState) -> dict:
    """Hard-stop path: session already escalated or allergen safety. Acknowledge only."""
    esc = state.get("_escalation_check", {})
    esc_type = esc.get("escalation_type", "default")

    from app.handoff import CUSTOMER_ESCALATION_MESSAGES
    template = CUSTOMER_ESCALATION_MESSAGES.get(esc_type, CUSTOMER_ESCALATION_MESSAGES["default"])
    order_id = state.get("order_id") or "N/A"
    customer_name = state.get("customer_name") or "Customer"
    msg = template.format(order_id=order_id, customer_name=customer_name)

    if esc_type == "allergen" and not state.get("escalated"):
        _do_escalation_db_writes(state, esc.get("reason", "Allergen safety"), esc_type)

    return {
        "final_response": msg,
        "messages": [AIMessage(content=msg)],
        "escalated": True,
    }


# ── Node: intake ──────────────────────────────────────────────────────────────

def intake_node(state: AgentState) -> dict:
    all_messages = state["messages"]
    last_human = next(
        (m for m in reversed(all_messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return {"resolution_type": "need_order_id"}

    user_text = last_human.content
    existing_order_id = state.get("order_id")

    extracted = extract_intake_context(user_text)

    # Build a short conversation context (last 6 messages) for the LLM
    recent = [m for m in all_messages if isinstance(m, (HumanMessage, AIMessage))][-6:]
    context_text = ""
    for m in recent[:-1]:  # all but the latest
        role = "Customer" if isinstance(m, HumanMessage) else "Agent"
        context_text += f"{role}: {m.content}\n"
    llm_user_content = (
        f"Conversation context:\n{context_text}\n" if context_text else ""
    ) + INTAKE_USER_TEMPLATE.format(message=user_text)

    try:
        if DEV_FAIL_LOUD:
            llm = _get_llm(IntakeOutput)
            result: IntakeOutput = llm.invoke([
                SystemMessage(content=INTAKE_SYSTEM),
                HumanMessage(content=llm_user_content),
            ])
        else:
            llm = _get_llm(IntakeOutput)
            result: IntakeOutput = llm.invoke([
                SystemMessage(content=INTAKE_SYSTEM),
                HumanMessage(content=llm_user_content),
            ])

        order_id = result.order_id or existing_order_id
        rag_context, policy_confidence = get_rag_context_with_confidence(
            f"{result.complaint_type}: {result.claim_description}"
        )

        allergens = list(set(
            (extracted.get("allergens_declared") or []) +
            (result.allergens_declared or [])
        ))
        dietary = extracted.get("dietary_restriction") or result.dietary_restriction
        is_human_req = result.is_explicit_human_request

        updates = {
            "order_id": order_id,
            "complaint_type": result.complaint_type,
            "claim_description": result.claim_description,
            "dietary_restriction": dietary,
            "allergens_declared": allergens,
            "customer_requested_resolution": result.customer_requested_resolution,
            "rag_context": rag_context,
            "policy_confidence": policy_confidence,
            "resolution_type": None if order_id else "need_order_id",
        }

        if is_human_req:
            updates["_escalation_check"] = {
                "should_escalate": True,
                "is_hard_stop": True,
                "reason": "Customer explicitly requested a human agent.",
                "escalation_type": "human_request",
            }

        return updates

    except Exception as e:
        logger.exception("intake_node LLM error")
        if DEV_FAIL_LOUD:
            raise
        # Scan all recent human messages for better keyword coverage
        human_texts = " ".join(
            m.content for m in all_messages[-12:] if isinstance(m, HumanMessage)
        )
        match = re.search(r"(ORD-\d+)", human_texts, re.IGNORECASE)
        order_id = match.group(1).upper() if match else existing_order_id
        t = human_texts.lower()
        if any(k in t for k in ("allerg", "peanut", "nut allerg", "dairy", "gluten")):
            ct = "allergen_violation"
        elif any(k in t for k in ("veg", "non-veg", "non veg", "vegetarian", "vegan", "halal", "kosher", "jain")):
            ct = "dietary_mismatch"
        elif any(k in t for k in ("late", "delay", "slow", "hours", "took too long")):
            ct = "late_delivery"
        elif any(k in t for k in ("missing", "didn't receive", "not in the bag", "wasn't there", "absent")):
            ct = "missing_item"
        elif any(k in t for k in ("wrong", "different food", "wrong item", "not what i ordered", "wrong order")):
            ct = "wrong_order"
        elif any(k in t for k in ("cold", "quality", "bad food", "stale", "undercooked", "spoiled", "raw")):
            ct = "quality_issue"
        elif any(k in t for k in ("free", "freebie", "complimentary", "promo item")):
            ct = "freebie_missing"
        elif any(k in t for k in ("payment", "charge", "double charged", "overcharged")):
            ct = "payment_issue"
        elif any(k in t for k in ("refund status", "where is my refund", "refund update")):
            ct = "refund_status"
        else:
            ct = "quality_issue"  # broad allowed options as safe default
        return {
            "order_id": order_id,
            "complaint_type": ct,
            "claim_description": user_text[:100],
            "allergens_declared": extracted.get("allergens_declared", []),
            "dietary_restriction": extracted.get("dietary_restriction"),
            "rag_context": "",
            "policy_confidence": 0.0,
            "resolution_type": None if order_id else "need_order_id",
        }


def route_after_intake(state: AgentState) -> str:
    esc = state.get("_escalation_check", {})
    if esc.get("should_escalate") and esc.get("is_hard_stop"):
        return "acknowledge"
    if state.get("resolution_type") == "need_order_id":
        return "request_info"
    return "verification"


# ── Node: verification ────────────────────────────────────────────────────────

def verification_node(state: AgentState) -> dict:
    """
    Cross-check verification between intake and adjudication.

    First pass (new complaint):
      - Photo type  → set case_status='awaiting_evidence', return upload prompt, END turn.
      - Timestamps  → run verify_timestamps(), set verification_result.
      - None type   → set verification_result = not_applicable.

    Second pass (resumed after evidence upload or skip):
      - Build VerificationResult from evidence_result (or inconclusive for skip).
      - Clear case_status, route to adjudication.
    """
    complaint_type  = state.get("complaint_type", "")
    action          = state.get("action")
    case_status_val = state.get("case_status")
    verifier_type   = get_verifier_type(complaint_type)

    # ── Resume: user uploaded a photo or skipped ───────────────────────────────
    if case_status_val == "awaiting_evidence":
        evidence_result = state.get("evidence_result")

        if action == "skip_evidence" or not evidence_result:
            ver = VerificationResult(
                status="inconclusive", method="photo",
                confidence=0.0,
                observations=["Customer skipped photo upload"],
                flags=["upload_skipped"],
            )
        elif complaint_type in ("dietary_mismatch", "wrong_order"):
            # Food-specific vision verification (menu-aware decision rules)
            ver = verify_food(
                state.get("order_details") or {},
                complaint_type,
                evidence_result,
                state.get("claim_description", ""),
            )
        else:
            # Generic photo evidence path (phash + EXIF + vision)
            ver = VerificationResult.from_evidence(evidence_result)

        _log_audit(
            state.get("session_id", ""), state.get("order_id"), state.get("customer_id"),
            "VERIFICATION_PHOTO",
            {"complaint_type": complaint_type, "status": ver.status,
             "flags": ver.flags, "confidence": ver.confidence},
        )
        return {
            "verification_result": ver.to_dict(),
            "case_status":         None,
            "evidence_required":   False,
            "action":              None,
        }

    # ── First pass: allergen claim → menu data only, no photo needed ───────────
    if verifier_type == "photo" and complaint_type == "dietary_mismatch":
        allergens_declared = state.get("allergens_declared") or []
        allergen_safety    = state.get("allergen_safety", False)
        if allergen_safety or allergens_declared:
            ver = verify_allergen(
                state.get("order_details") or {},
                allergens_declared,
            )
            _log_audit(
                state.get("session_id", ""), state.get("order_id"), state.get("customer_id"),
                "VERIFICATION_ALLERGEN",
                {"complaint_type": complaint_type, "status": ver.status,
                 "flags": ver.flags},
            )
            return {"verification_result": ver.to_dict(), "case_status": None}

    # ── First pass: photo type → request upload ────────────────────────────────
    if verifier_type == "photo":
        order_id     = state.get("order_id", "")
        customer_name = state.get("customer_name") or ""
        name_prefix  = f"Hi {customer_name}, " if customer_name else ""
        ct_display   = complaint_type.replace("_", " ")
        msg = (
            f"{name_prefix}to process your {ct_display} claim for order {order_id}, "
            "I need to verify the issue with a photo. Please upload a clear image showing "
            "the problem (JPEG, PNG or WebP, max 8 MB). You can tap 📷 or skip to see options."
        )
        pending_offer_upload = {
            "evidence_required": True,
            "resolution_type":   None,
            "compensation_amount": None,
            "order_id":          order_id,
            "customer_id":       state.get("customer_id"),
            "offer_text":        msg,
            "resolution_options": [],
            "created_at":        datetime.now(timezone.utc).isoformat(),
        }
        return {
            "case_status":     "awaiting_evidence",
            "final_response":  msg,
            "messages":        [AIMessage(content=msg)],
            "pending_offer":   pending_offer_upload,
            "evidence_required": True,
        }

    # ── First pass: timestamp type → verify delivery times ────────────────────
    if verifier_type == "timestamps":
        order_details = state.get("order_details")
        if not order_details:
            order_details = get_order_details(state.get("order_id") or "")
        ver = verify_timestamps(
            order_details or {},
            state.get("claim_description", ""),
            none_min=LATE_DELIVERY_NONE_MIN,
            low_min=LATE_DELIVERY_LOW_MIN,
            medium_min=LATE_DELIVERY_MEDIUM_MIN,
        )
        _log_audit(
            state.get("session_id", ""), state.get("order_id"), state.get("customer_id"),
            "VERIFICATION_TIMESTAMPS",
            {"complaint_type": complaint_type, "status": ver.status,
             "flags": ver.flags},
        )
        return {"verification_result": ver.to_dict(), "case_status": None}

    # ── First pass: none / not_applicable ─────────────────────────────────────
    return {
        "verification_result": VerificationResult.not_applicable().to_dict(),
        "case_status":         None,
    }


def route_after_verification(state: AgentState) -> str:
    if state.get("case_status") == "awaiting_evidence":
        return END   # waiting for photo upload — turn ends here
    return "adjudication"


# ── Node: request_info ────────────────────────────────────────────────────────

def request_info_node(state: AgentState) -> dict:
    return {
        "final_response": REQUEST_ORDER_ID_RESPONSE,
        "messages": [AIMessage(content=REQUEST_ORDER_ID_RESPONSE)],
    }


# ── Node: adjudication ────────────────────────────────────────────────────────

def adjudication_node(state: AgentState) -> dict:
    order_id          = state["order_id"]
    complaint_type    = state.get("complaint_type", "other")
    claim_description = state.get("claim_description", "")
    rag_context       = state.get("rag_context", "")
    policy_confidence = state.get("policy_confidence", 0.0)
    dietary_restrict  = state.get("dietary_restriction")
    allergens_declared = state.get("allergens_declared", [])

    # Collect previously rejected resolution types so the LLM avoids them
    offer_history = state.get("offer_history", [])
    rejected_resolutions = list({
        o["offer"]["resolution_type"]
        for o in offer_history
        if o.get("outcome") == "rejected"
    })

    # 1. Look up order
    order = get_order_details(order_id)
    if not order:
        return {
            "order_details": None,
            "resolution_type": "escalate",
            "reasoning": f"Order {order_id} not found.",
            "risk_score": 0,
            "risk_flags": ["Order not found"],
            "compensation_amount": None,
            "customer_id": None,
            "customer_name": None,
            "customer_email": None,
            "severity": "UNKNOWN",
            "allergen_safety": False,
        }

    customer_id    = order.get("customer_id", "")
    customer_name  = order.get("customer_name") or None  # None = unknown, not "Customer"
    customer_email = order.get("email", "")
    order_total    = float(order.get("total", 0))

    # 2. Parse items for severity classification
    items = order.get("items", [])
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except Exception:
            items = []

    # 3. Delivery delay — only computed when the complaint IS late_delivery
    delay_minutes = 0
    if complaint_type == "late_delivery":
        est = order.get("estimated_delivery_min") or 0
        act = order.get("actual_delivery_min") or 0
        delay_minutes = max(0, act - est)

    # 4. Severity classification
    severity, allergen_safety = classify_severity(
        complaint_type=complaint_type,
        claim_description=claim_description,
        items=items,
        dietary_restriction=dietary_restrict,
        allergens_declared=allergens_declared,
        delay_minutes=delay_minutes,
    )
    _log_audit(
        state.get("session_id", ""), order_id, order.get("customer_id"),
        "SEVERITY",
        {
            "complaint_type": complaint_type,
            "items_count": len(items),
            "dietary_restriction": dietary_restrict,
            "allergens_declared": allergens_declared,
            "delay_minutes": delay_minutes,
            "severity": severity,
            "allergen_safety": allergen_safety,
        },
    )

    # 5. Refund history + risk scoring
    history = get_refund_history(customer_id)
    history.same_issue_repeated = complaint_type in history.last_issues
    risk_score, risk_flags = compute_risk_score(history, order_total, order_total)

    if order_total > 100:
        risk_flags.append(f"High-value order: ${order_total:.2f}")

    # 6. Hard pre-emptions before LLM
    if order.get("status") == "refund":
        return {
            "order_details": order, "customer_id": customer_id,
            "customer_name": customer_name, "customer_email": customer_email,
            "resolution_type": "escalate",
            "reasoning": "Order already refunded. Possible duplicate claim.",
            "risk_score": risk_score, "risk_flags": risk_flags + ["Order already refunded"],
            "compensation_amount": None, "severity": severity, "allergen_safety": allergen_safety,
        }

    if allergen_safety and ALLERGEN_ALWAYS_ESCALATE:
        # Distinguish foreign_object critical from allergen safety for escalation messaging
        esc_type = (
            "foreign_object_critical"
            if complaint_type == "foreign_object"
            else "allergen"
        )
        return {
            "order_details": order, "customer_id": customer_id,
            "customer_name": customer_name, "customer_email": customer_email,
            "resolution_type": "escalate",
            "reasoning": (
                "Foreign object ingested/illness reported — immediate safety escalation."
                if esc_type == "foreign_object_critical"
                else "Allergen safety: customer declared allergen present in order."
            ),
            "risk_score": risk_score, "risk_flags": risk_flags + ["Safety escalation flag"],
            "compensation_amount": order_total, "severity": "CRITICAL", "allergen_safety": True,
            "_escalation_check": {
                "should_escalate": True, "is_hard_stop": True,
                "escalation_type": esc_type,
                "reason": "Safety escalation.",
            },
        }

    if risk_score > 50:
        risk_flags.append(f"Risk score {risk_score} exceeds threshold")

    if policy_confidence < POLICY_CONFIDENCE_MIN:
        risk_flags.append(f"Low policy confidence: {policy_confidence:.2f}")

    # 7. Resolution matrix: authoritative option list for (complaint_type, severity)
    matrix_options = get_options(complaint_type, severity)

    # CRITICAL / empty options → immediate escalation (allergen path already handled above)
    if not matrix_options:
        return {
            "order_details": order, "customer_id": customer_id,
            "customer_name": customer_name, "customer_email": customer_email,
            "resolution_type": "escalate",
            "reasoning": f"No options available for {complaint_type}/{severity}.",
            "risk_score": risk_score, "risk_flags": risk_flags,
            "compensation_amount": None, "severity": severity, "allergen_safety": allergen_safety,
        }

    # priority_handling guard: only valid when order is still in-progress
    order_status = order.get("status", "")
    if "priority_handling" in matrix_options and order_status not in ("preparing", "out_for_delivery"):
        matrix_options = [o for o in matrix_options if o != "priority_handling"]

    # The resolution_type stored in state is the first non-escalate option (for routing)
    non_esc = [o for o in matrix_options if o != "escalate"]
    resolution_type = non_esc[0] if non_esc else "escalate"

    # 8. Filter order data shown to LLM — only complaint-relevant fields
    order_context_for_llm = _filter_order_context(order, complaint_type)

    # 9. LLM used for risk analysis only — the matrix provides the resolution
    try:
        llm = _get_llm(AdjudicationOutput)
        user_prompt = ADJUDICATION_USER_TEMPLATE.format(
            complaint_type=complaint_type,
            severity=severity,
            claim_description=claim_description,
            customer_id=customer_id,
            order_details=json.dumps(order_context_for_llm, indent=2),
            dietary_restriction=dietary_restrict or "none",
            allergens_declared=", ".join(allergens_declared) or "none",
            count_7d=history.count_last_7_days,
            count_30d=history.count_last_30_days,
            lifetime=history.total_lifetime,
            last_issues=", ".join(history.last_issues) or "none",
            risk_score=risk_score,
            risk_flags=", ".join(risk_flags) or "none",
            policy_context=rag_context or "No specific policy retrieved.",
            policy_confidence=f"{policy_confidence:.2f}",
            allowed_resolutions=", ".join(matrix_options),
            rejected_resolutions=", ".join(rejected_resolutions) or "none",
            max_compensation="see options",
            compensation_form="varies",
        )
        result: AdjudicationOutput = llm.invoke([
            SystemMessage(content=ADJUDICATION_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        all_flags = list(set(risk_flags + result.risk_flags))
        # LLM resolution_type is IGNORED — matrix is authoritative

    except Exception as e:
        logger.exception("adjudication_node LLM error — using deterministic matrix options")
        if DEV_FAIL_LOUD:
            raise
        all_flags = risk_flags + [f"LLM adjudication error ({type(e).__name__}) — deterministic"]
        result = type("R", (), {"reasoning": f"LLM unavailable. Matrix: {matrix_options}.", "risk_flags": []})()

    # ── Apply verification-layer severity adjustments ──────────────────────────
    ver = state.get("verification_result") or {}
    ver_status = ver.get("status", "not_applicable")
    ver_method = ver.get("method", "none")
    ver_flags  = ver.get("flags", [])

    # Photo contradicted or duplicate → escalate immediately (spec: never auto-deny)
    if ver_method == "photo" and ver_status == "contradicted":
        return {
            "order_details": order, "customer_id": customer_id,
            "customer_name": customer_name, "customer_email": customer_email,
            "resolution_type": "escalate",
            "reasoning": "Photo evidence contradicted or was a duplicate — escalating for human review.",
            "risk_score": risk_score, "risk_flags": all_flags + ["photo_contradicted"],
            "compensation_amount": None, "severity": severity, "allergen_safety": allergen_safety,
            "resolution_attempts": state.get("resolution_attempts", 0) + 1,
            "_escalation_check": {
                "should_escalate": True, "is_hard_stop": False,
                "escalation_type": "evidence_rejected",
                "reason": "Photo evidence inconsistent or previously seen.",
            },
        }

    # Photo inconclusive → cap severity one tier lower + ensure escalate button visible
    if ver_method == "photo" and ver_status == "inconclusive":
        severity = _lower_severity(severity)
        all_flags.append("photo_inconclusive_severity_capped")

    # Timestamps contradicted → mark for factual message in offer_node
    if ver_method == "timestamps" and ver_status == "contradicted":
        all_flags.append("timestamp_claim_contradicted")

    # Timestamps verified → use data-derived severity from flags
    if ver_method == "timestamps" and ver_status == "verified":
        for flag in ver_flags:
            if flag.startswith("data_severity:"):
                data_sev = flag.split(":", 1)[1].upper()
                if data_sev in ("HIGH", "MEDIUM", "LOW"):
                    severity = data_sev
                break

    # Re-derive matrix options with the potentially adjusted severity
    matrix_options = get_options(complaint_type, severity) or matrix_options
    if "priority_handling" in matrix_options and order_status not in ("preparing", "out_for_delivery"):
        matrix_options = [o for o in matrix_options if o != "priority_handling"]
    non_esc = [o for o in matrix_options if o != "escalate"]
    resolution_type = non_esc[0] if non_esc else "escalate"

    # Verified foreign_object → log restaurant hygiene flag
    if complaint_type == "foreign_object" and ver_status == "verified":
        all_flags.append("restaurant_hygiene_flag")
        _log_audit(
            state.get("session_id", ""), order_id, customer_id,
            "HYGIENE_FLAG",
            {"complaint_type": complaint_type, "order_id": order_id,
             "verification_confidence": ver.get("confidence", 0)},
        )

    # ── Compute final amount ───────────────────────────────────────────────────
    final_amount = compute_option_amount(resolution_type, order_total, complaint_type, severity)

    _log_audit(
        state.get("session_id", ""), order_id, customer_id,
        "ADJUDICATION",
        {"resolution_type": resolution_type, "severity": severity,
         "risk_score": risk_score, "amount": final_amount,
         "complaint_type": complaint_type,
         "verification_status": ver_status, "verification_method": ver_method},
    )

    return {
        "order_details": order,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "customer_email": customer_email,
        "resolution_type": resolution_type,
        "compensation_amount": final_amount,
        "reasoning": result.reasoning if hasattr(result, "reasoning") else "",
        "risk_score": risk_score,
        "risk_flags": all_flags,
        "severity": severity,
        "allergen_safety": allergen_safety,
        "resolution_attempts": state.get("resolution_attempts", 0) + 1,
    }


def route_after_adjudication(state: AgentState) -> str:
    rt = state.get("resolution_type", "")
    if rt == "escalate" or state.get("allergen_safety") and ALLERGEN_ALWAYS_ESCALATE:
        return "escalate"
    return "offer"


def route_after_offer(state: AgentState) -> str:
    """Route to escalate when offer_node detected bad evidence; otherwise END."""
    if state.get("resolution_type") == "escalate":
        return "escalate"
    return END


# ── Node: offer ───────────────────────────────────────────────────────────────

def offer_node(state: AgentState) -> dict:
    """Generate an offer message and present ALL matrix options as buttons."""
    complaint_type      = state.get("complaint_type", "other")
    severity            = state.get("severity", "LOW")
    order_id            = state.get("order_id", "")
    customer_name       = state.get("customer_name") or ""
    claim_description   = state.get("claim_description", "")
    offer_history       = state.get("offer_history", [])
    correction_detected = state.get("correction_detected", False)
    dissatisfaction_count = state.get("dissatisfaction_count", 0)
    order_total         = float((state.get("order_details") or {}).get("total", 0))
    ver                 = state.get("verification_result") or {}
    ver_method          = ver.get("method", "none")
    ver_status          = ver.get("status", "not_applicable")
    risk_flags          = list(state.get("risk_flags") or [])

    # ── Timestamp claim contradicts data → factual explanation + human option ─
    if ver_method == "timestamps" and ver_status == "contradicted" and "timestamp_claim_contradicted" in risk_flags:
        obs_lines = ver.get("observations", [])
        factual   = " ".join(obs_lines) or "Our records do not show a significant delivery delay."
        msg = (
            f"I've reviewed the delivery records for order {order_id}. "
            f"{factual} "
            "If you believe our records are incorrect, a human agent can look into this further."
        )
        resolution_options = [{"type": "escalate", "label": "Talk to a Human Agent", "amount": None}]
        pending_offer = {
            "resolution_type": "escalate",
            "compensation_amount": None,
            "order_id": order_id,
            "customer_id": state.get("customer_id"),
            "offer_text": msg,
            "resolution_options": resolution_options,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return {
            "final_response": msg,
            "messages": [AIMessage(content=msg)],
            "pending_offer": pending_offer,
            "resolution_type": "escalate",
        }

    # Build option list — add escalate for MEDIUM/LOW after first dissatisfaction
    if dissatisfaction_count > 0:
        all_options = get_options_after_dissatisfaction(complaint_type, severity)
    else:
        all_options = get_options(complaint_type, severity)

    # Filter out options already executed this session
    executed = {
        o["offer"]["resolution_type"]
        for o in offer_history
        if o.get("outcome") == "accepted"
    }
    all_options = [o for o in all_options if o not in executed]

    if not all_options:
        all_options = ["escalate"]

    # Photo inconclusive (severity capped) → always show escalate button as an exit
    if ver_method == "photo" and ver_status == "inconclusive":
        if "escalate" not in all_options:
            all_options = all_options + ["escalate"]

    # Compute amount for each option
    resolution_options = []
    for opt in all_options:
        amount = compute_option_amount(opt, order_total, complaint_type, severity)
        label = _resolution_label(opt)
        if amount is not None:
            label = f"{label} (${amount:.2f})"
        resolution_options.append({"type": opt, "label": label, "amount": amount})

    # Primary resolution for state tracking (first non-escalate)
    primary_rt = next((o["type"] for o in resolution_options if o["type"] != "escalate"), "escalate")
    primary_amount = next((o["amount"] for o in resolution_options if o["type"] == primary_rt), None)

    correction_note = (
        "Customer corrected an earlier misclassification — the previous issue type was wrong."
        if correction_detected else ""
    )

    # LLM writes only the intro/apology text — options are shown as buttons
    try:
        llm = _get_llm()
        user_prompt = OFFER_USER_TEMPLATE.format(
            customer_name=customer_name,
            complaint_type=complaint_type,
            severity=severity,
            order_id=order_id,
            claim_description=claim_description,
            correction_note=correction_note,
        )
        response = llm.invoke([
            SystemMessage(content=OFFER_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        offer_text = response.content

    except Exception as e:
        logger.exception("offer_node LLM error")
        if DEV_FAIL_LOUD:
            raise
        name_prefix = f"Hi {customer_name}, " if customer_name else ""
        if severity == "HIGH":
            offer_text = f"{name_prefix}I sincerely apologise for this serious issue with your order {order_id}. Please choose how you'd like this resolved:"
        elif severity == "MEDIUM":
            offer_text = f"{name_prefix}I'm sorry about the problem with your order {order_id}. Here are your options:"
        else:
            offer_text = f"{name_prefix}Sorry about that with order {order_id}. Here's what I can offer:"
        if correction_note:
            offer_text = f"I apologise for the earlier mix-up. {offer_text}"

    pending_offer = {
        "resolution_type": primary_rt,
        "compensation_amount": primary_amount,
        "order_id": order_id,
        "customer_id": state.get("customer_id"),
        "offer_text": offer_text,
        "resolution_options": resolution_options,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _log_audit(
        state.get("session_id", ""), order_id, state.get("customer_id"),
        "OFFER",
        {"options": [o["type"] for o in resolution_options], "severity": severity,
         "dissatisfaction_count": dissatisfaction_count, "complaint_type": complaint_type},
    )

    return {
        "final_response": offer_text,
        "messages": [AIMessage(content=offer_text)],
        "pending_offer": pending_offer,
        "resolution_type": primary_rt,
        "compensation_amount": primary_amount,
        "correction_detected": False,  # consumed — clear for next turn
    }


# ── Node: confirm ─────────────────────────────────────────────────────────────

def confirm_node(state: AgentState) -> dict:
    """Detect whether the customer is accepting, rejecting, or correcting/raising a new complaint."""
    pending_offer = state.get("pending_offer", {}) or {}
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    user_text = last_human.content if last_human else ""

    # ── 1. Deterministic path: frontend button sent an explicit action ─────────
    action = state.get("action")

    if action == "accept":
        return {
            "confirm_response": "accept",
            "action": None,
            "selected_resolution_type": None,
        }

    if action == "select":
        sel = state.get("selected_resolution_type") or ""
        valid_types = [o["type"] for o in (pending_offer.get("resolution_options") or [])]

        if sel == "escalate":
            # Customer chose "Talk to a Human Agent" from the picker
            return {
                "confirm_response": "human_request",
                "action": None,
                "selected_resolution_type": None,
            }

        if sel in valid_types:
            # Use amount from the resolution_options list (already computed by offer_node)
            opt_entry = next(
                (o for o in (pending_offer.get("resolution_options") or []) if o["type"] == sel),
                None,
            )
            new_amount = opt_entry["amount"] if opt_entry else compute_option_amount(
                sel,
                float((state.get("order_details") or {}).get("total", 0)),
                state.get("complaint_type", "other"),
                state.get("severity", "LOW"),
            )
            updated_offer = {**pending_offer, "resolution_type": sel, "compensation_amount": new_amount}
            return {
                "confirm_response": "accept",
                "pending_offer": updated_offer,
                "resolution_type": sel,
                "compensation_amount": new_amount,
                "action": None,
                "selected_resolution_type": None,
            }
        # Invalid selection → treat as dissatisfied
        return {"confirm_response": "dissatisfied", "action": None, "selected_resolution_type": None}

    # ── 2. LLM-based classification (text message, no button action) ───────────
    try:
        llm = _get_llm(ConfirmOutput)
        result: ConfirmOutput = llm.invoke([
            SystemMessage(content=CONFIRM_SYSTEM.format(
                offer_text=pending_offer.get("offer_text", "")
            )),
            HumanMessage(content=CONFIRM_USER_TEMPLATE.format(message=user_text)),
        ])
        response_type = result.response_type
    except Exception:
        logger.exception("confirm_node LLM error — falling back to keyword detection")
        norm = user_text.lower()
        if any(w in norm for w in ["yes", "ok", "sure", "proceed", "go ahead", "agree", "accept", "please"]):
            response_type = "accept"
        elif any(w in norm for w in ["no", "not enough", "want more", "refuse", "dissatisfied", "unhappy",
                                      "not happy", "something else", "don't like", "none of these"]):
            response_type = "dissatisfied"
        else:
            response_type = "new_complaint"

    updates: dict = {"confirm_response": response_type}

    if response_type == "dissatisfied":
        # Increment count; offer_node will add escalate for MEDIUM/LOW on re-offer
        updates["dissatisfaction_count"] = state.get("dissatisfaction_count", 0) + 1
        updates["pending_offer"] = None  # clear so offer_node rebuilds with full options

    elif response_type == "new_complaint":
        is_correction = _is_correction(user_text)

        # Save the unresolved complaint into open_issues before wiping state
        open_issues = list(state.get("open_issues") or [])
        if state.get("complaint_type"):
            open_issues.append({
                "complaint_type": state.get("complaint_type"),
                "claim_description": state.get("claim_description"),
                "severity": state.get("severity"),
                "order_id": state.get("order_id"),
                "offer_history": list(state.get("offer_history") or []),
                "resolution_type": None,
            })
        updates["open_issues"] = open_issues
        updates["pending_offer"] = None

        # Preserve order_id unless the new message explicitly names a different order
        new_order_match = re.search(r"\b(ORD-\d+)\b", user_text, re.IGNORECASE)
        if new_order_match:
            new_oid = new_order_match.group(1).upper()
            if new_oid != (state.get("order_id") or "").upper():
                updates["order_id"] = new_oid
        # (If no different order mentioned, order_id is intentionally kept)

        # Reset all complaint-specific state
        updates["complaint_type"]        = None
        updates["claim_description"]     = None
        updates["severity"]              = None
        updates["allergen_safety"]       = False
        updates["resolution_type"]       = None
        updates["compensation_amount"]   = None
        updates["dietary_restriction"]   = None
        updates["allergens_declared"]    = []
        updates["resolution_attempts"]   = 0
        updates["offer_history"]         = []
        updates["dissatisfaction_count"] = 0  # reset for new complaint
        updates["correction_detected"]   = is_correction

    return updates


def route_after_confirm(state: AgentState) -> str:
    cr = state.get("confirm_response")
    if cr == "accept":
        return "execute"
    elif cr == "human_request":
        return "escalate"
    elif cr == "new_complaint":
        return "intake"
    elif cr == "dissatisfied":
        # Second (or more) typed dissatisfaction → escalate
        if state.get("dissatisfaction_count", 0) >= 2:
            return "escalate"
        return "offer"
    else:
        return "offer"


# ── Node: execute ─────────────────────────────────────────────────────────────

def execute_node(state: AgentState) -> dict:
    """Execute the accepted pending_offer — write to DB and generate confirmation."""
    pending_offer = state.get("pending_offer", {})
    resolution_type     = pending_offer.get("resolution_type", "apology")
    compensation_amount = pending_offer.get("compensation_amount")
    order_id            = pending_offer.get("order_id") or state.get("order_id", "")
    customer_id         = pending_offer.get("customer_id") or state.get("customer_id")
    customer_name       = state.get("customer_name") or ""
    customer_email      = state.get("customer_email", "")
    form                = pending_offer.get("compensation_form", "none")
    claim_description   = state.get("claim_description", "")

    # Execute DB actions
    if resolution_type == "full_refund":
        set_refund_status(order_id)
    elif resolution_type == "partial_refund" and compensation_amount:
        set_partial_refund_status(order_id, compensation_amount)
    elif resolution_type == "coupon":
        send_voucher_email(
            customer_email, customer_name or "Customer",
            f"${compensation_amount:.2f}" if compensation_amount else "5%",
            claim_description,
        )
    elif resolution_type == "store_credit":
        set_partial_refund_status(order_id, compensation_amount or 0)

    try:
        from app.tools import make_action_id
        with _conn() as conn:
            action_id = make_action_id(conn)
            conn.execute(
                "INSERT INTO support_actions "
                "(action_id, customer_id, order_id, action_type, amount, reason, status, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    action_id, customer_id, order_id,
                    resolution_type.upper(),
                    compensation_amount,
                    claim_description,
                    "COMPLETED",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except Exception:
        pass

    amount_display = "N/A"
    if compensation_amount is not None:
        if form == "coupon":
            amount_display = f"${compensation_amount:.2f} coupon"
        elif form == "credit":
            amount_display = f"${compensation_amount:.2f} store credit"
        else:
            amount_display = f"${compensation_amount:.2f}"

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=EXECUTE_RESPONSE_SYSTEM),
            HumanMessage(content=EXECUTE_USER_TEMPLATE.format(
                customer_name=customer_name,
                resolution_type=resolution_type,
                amount_display=amount_display,
                order_id=order_id,
            )),
        ])
        final_response = response.content
    except Exception as e:
        logger.exception("execute_node LLM error")
        if DEV_FAIL_LOUD:
            raise
        name_prefix = f"Done, {customer_name}! " if customer_name else "Done! "
        if resolution_type == "full_refund":
            final_response = f"{name_prefix}Your full refund of {amount_display} for order {order_id} has been processed and will appear in 3–5 business days."
        elif resolution_type in ("partial_refund", "store_credit"):
            final_response = f"{name_prefix}A {amount_display} refund for order {order_id} has been processed and will appear in 3–5 business days."
        elif resolution_type == "coupon":
            final_response = f"{name_prefix}A {amount_display} voucher has been sent to your email. Thank you for your patience!"
        else:
            final_response = f"Your request for order {order_id} has been processed. Thank you!"

    _log_audit(
        state.get("session_id", ""), order_id, customer_id,
        "EXECUTION",
        {"resolution_type": resolution_type, "amount": compensation_amount},
    )

    offer_history = list(state.get("offer_history", []))
    offer_history.append({
        "offer": pending_offer,
        "outcome": "accepted",
        "turn": len(state["messages"]),
    })

    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
        "pending_offer": None,
        "offer_history": offer_history,
        "dissatisfaction_count": 0,
    }


# ── Node: escalate ────────────────────────────────────────────────────────────

def _do_escalation_db_writes(state: dict, reason: str, esc_type: str) -> None:
    order_id = state.get("order_id", "")
    if order_id:
        set_humancheck_status(order_id)
    log_escalation(
        session_id=state.get("session_id", ""),
        order_id=order_id,
        reason=reason,
        conversation=[
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in state.get("messages", [])
        ],
        order_details=state.get("order_details") or {},
        risk_score=state.get("risk_score", 0),
        risk_flags=state.get("risk_flags", []),
        suggested_action=esc_type,
    )


def escalate_node(state: AgentState) -> dict:
    esc_check = state.get("_escalation_check", {})
    esc_type  = esc_check.get("escalation_type", "default")
    reason    = esc_check.get("reason", state.get("reasoning", "Escalation required."))

    if not esc_type and state.get("allergen_safety"):
        esc_type = "allergen"
    elif not esc_type:
        esc_type = "max_attempts"

    # Only write DB escalation if not already escalated (prevents duplicates)
    if not state.get("escalated"):
        _do_escalation_db_writes(state, reason, esc_type)

    try:
        llm_callable = _get_llm()
        agent_card, customer_msg = generate_handoff_summary(state, reason, esc_type, llm_callable)
    except Exception:
        from app.handoff import CUSTOMER_ESCALATION_MESSAGES
        template = CUSTOMER_ESCALATION_MESSAGES.get(esc_type, CUSTOMER_ESCALATION_MESSAGES["default"])
        customer_msg = template.format(
            order_id=state.get("order_id") or "N/A",
            customer_name=state.get("customer_name") or "Customer",
        )

    ev = state.get("evidence_result")
    evidence_audit = {}
    if ev:
        evidence_audit = {
            "evidence_verdict": ev.get("verdict"),
            "evidence_confidence": ev.get("confidence"),
            "evidence_observations": ev.get("observations", []),
            "evidence_duplicate": ev.get("duplicate_image", False),
            "evidence_thumbnail": ev.get("thumbnail_path", ""),
        }

    _log_audit(
        state.get("session_id", ""), state.get("order_id"), state.get("customer_id"),
        "ESCALATION",
        {"escalation_type": esc_type, "reason": reason,
         "complaint_type": state.get("complaint_type", "unknown"),
         **evidence_audit},
    )

    return {
        "final_response": customer_msg,
        "messages": [AIMessage(content=customer_msg)],
        "escalated": True,
        "pending_offer": None,
    }


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("escalation_check", escalation_check_node)
    g.add_node("acknowledge",      acknowledge_node)
    g.add_node("intake",           intake_node)
    g.add_node("verification",     verification_node)
    g.add_node("request_info",     request_info_node)
    g.add_node("adjudication",     adjudication_node)
    g.add_node("offer",            offer_node)
    g.add_node("confirm",          confirm_node)
    g.add_node("execute",          execute_node)
    g.add_node("escalate",         escalate_node)

    g.set_entry_point("escalation_check")

    g.add_conditional_edges(
        "escalation_check",
        route_after_escalation_check,
        {
            "acknowledge":  "acknowledge",
            "confirm":      "confirm",
            "intake":       "intake",
            "verification": "verification",
        },
    )

    g.add_edge("acknowledge", END)

    g.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "acknowledge":  "acknowledge",
            "request_info": "request_info",
            "verification": "verification",
        },
    )

    g.add_edge("request_info", END)

    g.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "adjudication": "adjudication",
            END:            END,
        },
    )

    g.add_conditional_edges(
        "adjudication",
        route_after_adjudication,
        {
            "escalate": "escalate",
            "offer":    "offer",
        },
    )

    g.add_conditional_edges(
        "offer",
        route_after_offer,
        {
            "escalate": "escalate",
            END:        END,
        },
    )

    g.add_conditional_edges(
        "confirm",
        route_after_confirm,
        {
            "execute":      "execute",
            "intake":       "intake",
            "adjudication": "adjudication",
            "escalate":     "escalate",
            "offer":        "offer",
        },
    )

    g.add_edge("execute",  END)
    g.add_edge("escalate", END)

    return g.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
