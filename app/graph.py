"""
LangGraph three-node pipeline: Intake → Adjudication → Action

[intake_node]
    ↓ (has order_id?)
    ├── NO  → [request_info_node] → END
    └── YES → [adjudication_node]
                    ↓
              [action_node] → END

Each node is a pure function on AgentState.
Token cost per turn is logged to stdout when DEBUG=True in .env.
"""
import json
import os
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.config import GROQ_API_KEY, LLM_MODEL, LLM_PROVIDER, OPENAI_API_KEY
from app.prompts import (
    INTAKE_SYSTEM, INTAKE_USER_TEMPLATE,
    ADJUDICATION_SYSTEM, ADJUDICATION_USER_TEMPLATE,
    RESPONSE_SYSTEM, RESPONSE_USER_TEMPLATE,
    REQUEST_ORDER_ID_RESPONSE,
)
from app.rag import get_rag_context
from app.risk import compute_risk_score, RefundHistory
from app.tools import (
    get_order_details, get_refund_history,
    set_refund_status, set_partial_refund_status,
    set_humancheck_status, log_escalation,
    send_voucher_email,
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    order_id: Optional[str]
    customer_id: Optional[str]
    customer_name: Optional[str]
    customer_email: Optional[str]
    complaint_type: Optional[str]
    claim_description: Optional[str]
    order_details: Optional[dict]
    risk_score: int
    risk_flags: list[str]
    decision: Optional[str]
    refund_amount: Optional[float]
    reasoning: Optional[str]
    rag_context: Optional[str]
    final_response: Optional[str]
    token_log: list[dict]


# ── Pydantic schemas for structured LLM output ───────────────────────────────

class IntakeOutput(BaseModel):
    complaint_type: str = Field(description="One of: missing_item, quality_issue, wrong_order, late_delivery, payment_issue, refund_status, other")
    order_id: Optional[str] = Field(None, description="Order ID if mentioned, else null")
    claim_description: str = Field(description="One-sentence summary of the complaint")
    confidence: float = Field(ge=0.0, le=1.0)


class AdjudicationOutput(BaseModel):
    decision: str = Field(description="One of: full_refund, partial_refund, deny, escalate")
    refund_amount: Optional[float] = Field(None, description="Refund in dollars if partial or full")
    reasoning: str = Field(description="Internal reasoning, max 100 words")
    risk_flags: list[str] = Field(default_factory=list)


# ── LLM factory ──────────────────────────────────────────────────────────────

def _get_llm(structured_output_class=None):
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)

    if structured_output_class:
        return llm.with_structured_output(structured_output_class)
    return llm


# ── Node 1: Intake ────────────────────────────────────────────────────────────

def intake_node(state: AgentState) -> dict:
    """
    Classify the complaint and extract order_id.
    Uses structured output for deterministic JSON parsing.
    """
    # Get last human message
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return {"decision": "need_order_id"}

    user_text = last_human.content

    # If we already have an order_id from a previous turn, keep it
    existing_order_id = state.get("order_id")

    try:
        llm = _get_llm(IntakeOutput)
        from langchain_core.messages import SystemMessage
        result: IntakeOutput = llm.invoke([
            SystemMessage(content=INTAKE_SYSTEM),
            HumanMessage(content=INTAKE_USER_TEMPLATE.format(message=user_text)),
        ])

        order_id = result.order_id or existing_order_id

        # Retrieve RAG context based on complaint type
        rag_context = get_rag_context(f"{result.complaint_type}: {result.claim_description}")

        return {
            "order_id": order_id,
            "complaint_type": result.complaint_type,
            "claim_description": result.claim_description,
            "rag_context": rag_context,
            "decision": None if order_id else "need_order_id",
        }

    except Exception as e:
        # Fallback: try to find ORD-\d+ in text
        import re
        match = re.search(r"(ORD-\d+)", user_text, re.IGNORECASE)
        order_id = match.group(1).upper() if match else existing_order_id
        return {
            "order_id": order_id,
            "complaint_type": "other",
            "claim_description": user_text[:100],
            "rag_context": "",
            "decision": None if order_id else "need_order_id",
        }


# ── Node 2: Adjudication ──────────────────────────────────────────────────────

def adjudication_node(state: AgentState) -> dict:
    """
    Core reasoning node: look up order, score risk, call RISE-M prompt.
    This is where the agent decides: full_refund / partial_refund / deny / escalate.
    """
    order_id = state["order_id"]
    complaint_type = state.get("complaint_type", "other")
    claim_description = state.get("claim_description", "")
    rag_context = state.get("rag_context", "")

    # 1. Look up order from DB
    order = get_order_details(order_id)
    if not order:
        return {
            "order_details": None,
            "decision": "escalate",
            "reasoning": f"Order {order_id} not found in system.",
            "risk_score": 0,
            "risk_flags": ["Order not found"],
            "refund_amount": None,
            "customer_id": None,
            "customer_name": "Customer",
            "customer_email": None,
        }

    customer_id = order.get("customer_id", "")
    customer_name = order.get("customer_name", "Customer")
    customer_email = order.get("email", "")

    # 2. Get refund history
    history = get_refund_history(customer_id)

    # Mark same_issue_repeated if this complaint type was used before
    history.same_issue_repeated = complaint_type in history.last_issues

    # 3. Estimate claim amount (full order value as proxy if unknown)
    order_total = float(order.get("total", 0))
    claim_amount = order_total  # worst case — adjudication may narrow it

    # 4. Risk scoring
    risk_score, risk_flags = compute_risk_score(history, claim_amount, order_total)

    # 5. Hard escalation triggers (before LLM call — cheap to evaluate)
    if order.get("status") == "refund":
        return {
            "order_details": order,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "decision": "escalate",
            "reasoning": "Order has already been refunded. Possible duplicate claim.",
            "risk_score": risk_score,
            "risk_flags": risk_flags + ["Order already refunded"],
            "refund_amount": None,
        }

    if order_total > 100:
        risk_flags.append(f"High-value order: ${order_total:.2f}")

    # 6. RISE-M adjudication LLM call
    try:
        llm = _get_llm(AdjudicationOutput)
        from langchain_core.messages import SystemMessage

        user_prompt = ADJUDICATION_USER_TEMPLATE.format(
            complaint_type=complaint_type,
            claim_description=claim_description,
            customer_name=customer_name,
            customer_id=customer_id,
            order_details=json.dumps(order, indent=2),
            count_7d=history.count_last_7_days,
            count_30d=history.count_last_30_days,
            lifetime=history.total_lifetime,
            last_issues=", ".join(history.last_issues) or "none",
            risk_score=risk_score,
            risk_flags=", ".join(risk_flags) or "none",
            policy_context=rag_context or "No specific policy retrieved.",
        )

        result: AdjudicationOutput = llm.invoke([
            SystemMessage(content=ADJUDICATION_SYSTEM),
            HumanMessage(content=user_prompt),
        ])

        # Merge LLM risk flags with our computed ones
        all_flags = list(set(risk_flags + result.risk_flags))

        return {
            "order_details": order,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "decision": result.decision,
            "refund_amount": result.refund_amount,
            "reasoning": result.reasoning,
            "risk_score": risk_score,
            "risk_flags": all_flags,
        }

    except Exception as e:
        # On LLM failure, escalate rather than guess
        return {
            "order_details": order,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "decision": "escalate",
            "reasoning": f"Adjudication failed: {e}. Escalating for safety.",
            "risk_score": risk_score,
            "risk_flags": risk_flags + ["LLM adjudication error"],
            "refund_amount": None,
        }


# ── Node 3: Action ────────────────────────────────────────────────────────────

def action_node(state: AgentState) -> dict:
    """
    Execute the decision and generate a customer-facing response.
    Writes to DB, logs escalations, sends vouchers.
    """
    decision = state.get("decision")
    order_id = state.get("order_id", "")
    refund_amount = state.get("refund_amount")
    reasoning = state.get("reasoning", "")
    customer_name = state.get("customer_name", "Customer")
    customer_email = state.get("customer_email", "")
    claim_description = state.get("claim_description", "")
    order_details = state.get("order_details") or {}
    risk_score = state.get("risk_score", 0)
    risk_flags = state.get("risk_flags", [])
    session_id = state.get("session_id", "")

    # Execute the decision
    if decision == "full_refund":
        set_refund_status(order_id)
        refund_amount = refund_amount or float(order_details.get("total", 0))

    elif decision == "partial_refund" and refund_amount:
        set_partial_refund_status(order_id, refund_amount)

    elif decision == "escalate":
        set_humancheck_status(order_id)
        # Build conversation log for handoff record
        conversation = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in state.get("messages", [])
        ]
        log_escalation(
            session_id=session_id,
            order_id=order_id,
            reason=reasoning,
            conversation=conversation,
            order_details=order_details,
            risk_score=risk_score,
            risk_flags=risk_flags,
        )

    elif decision == "deny":
        pass  # No DB change needed

    # Generate customer-facing response via LLM
    try:
        llm = _get_llm()
        from langchain_core.messages import SystemMessage

        amt_display = f"${refund_amount:.2f}" if refund_amount is not None else "N/A"
        user_prompt = RESPONSE_USER_TEMPLATE.format(
            customer_name=customer_name,
            decision=decision,
            refund_amount=amt_display,
            order_id=order_id,
            reasoning=reasoning or "",
            claim_description=claim_description or "",
        )

        response_msg = llm.invoke([
            SystemMessage(content=RESPONSE_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        final_response = response_msg.content

    except Exception as llm_err:
        # Fallback responses — evaluated lazily to avoid NoneType format errors
        amt_str = f"${refund_amount:.2f}" if refund_amount is not None else "the applicable amount"
        if decision == "full_refund":
            final_response = f"I'm sorry about the issue with your order {order_id}. A full refund has been processed and will appear in 3–5 business days."
        elif decision == "partial_refund":
            final_response = f"I've processed a partial refund of {amt_str} for your order {order_id}. It will appear in 3–5 business days."
        elif decision == "deny":
            final_response = f"After reviewing your order {order_id}, we were unable to approve this claim. {reasoning or 'Please contact support for details.'}"
        elif decision == "escalate":
            final_response = f"I've escalated your case for {order_id} to our review team. They'll follow up with you within 24 hours."
        elif decision == "need_order_id":
            final_response = REQUEST_ORDER_ID_RESPONSE
        else:
            final_response = "I'm reviewing your case and will get back to you shortly."
        # Surface API key / auth errors clearly for the developer
        if "invalid_api_key" in str(llm_err).lower() or "authentication" in str(llm_err).lower() or "401" in str(llm_err):
            final_response = "⚠️ AI service is unavailable (invalid API key). Please check your GROQ_API_KEY in .env."

    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
    }


# ── Need-info node (short-circuit when no order ID) ──────────────────────────

def request_info_node(state: AgentState) -> dict:
    return {
        "final_response": REQUEST_ORDER_ID_RESPONSE,
        "messages": [AIMessage(content=REQUEST_ORDER_ID_RESPONSE)],
    }


# ── Conditional routing ───────────────────────────────────────────────────────

def route_after_intake(state: AgentState) -> str:
    if state.get("decision") == "need_order_id":
        return "request_info"
    return "adjudication"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intake", intake_node)
    g.add_node("request_info", request_info_node)
    g.add_node("adjudication", adjudication_node)
    g.add_node("action", action_node)

    g.set_entry_point("intake")

    g.add_conditional_edges(
        "intake",
        route_after_intake,
        {"request_info": "request_info", "adjudication": "adjudication"},
    )
    g.add_edge("request_info", END)
    g.add_edge("adjudication", "action")
    g.add_edge("action", END)

    return g.compile()


# Singleton compiled graph
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
