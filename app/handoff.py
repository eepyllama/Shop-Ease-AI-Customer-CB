"""
Human handoff summary generator.

Produces two outputs for each escalation:
  1. A structured dict for ops_tools / the agent dashboard.
  2. A short customer-facing message acknowledging the escalation.

The LLM writes a concise agent card (5–7 bullet points) summarising the case
so the human agent can act without reading the full conversation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

HANDOFF_SYSTEM = """# ROLE
You are a support quality analyst writing a handoff card for a human support agent.

# INSTRUCTIONS
Summarise the customer's case in 5–7 concise bullet points.
The agent has no context — give them everything they need to act immediately.

# STEPS
1. State the complaint type and order ID.
2. Describe what the customer says happened.
3. Note any allergen or dietary flag if present.
4. State the risk score and key risk flags.
5. State what resolution (if any) was offered and whether the customer accepted.
6. State the escalation reason.
7. Recommend next action (refund, redelivery, verify evidence, etc.).

# EXPECTATIONS
- Under 150 words total.
- Bullet list format.
- Do NOT address the customer — this is for the agent.
- Do NOT hallucinate details not provided.
"""

CUSTOMER_ESCALATION_MESSAGES = {
    "allergen": (
        "IMPORTANT: Please do not consume any items from this order. "
        "I've flagged this as a food safety concern and escalated it to our specialist team. "
        "They will contact you within 2 hours to arrange a full refund and follow up."
    ),
    "human_request": (
        "Absolutely — I'm connecting you with a human agent right now. "
        "A member of our team will follow up with you within 24 hours. "
        "Your case reference is {order_id}."
    ),
    "max_attempts": (
        "I want to make sure you get the right resolution for this. "
        "I'm escalating your case to our senior support team. "
        "They'll review everything and get back to you within 24 hours."
    ),
    "max_rejections": (
        "I understand none of the options I've offered have felt right. "
        "I'm escalating your case to a human agent who can work directly with you on a solution. "
        "You'll hear from them within 24 hours."
    ),
    "loop_detected": (
        "It looks like we've been going around in circles — I'm sorry for the frustration. "
        "I'm handing your case to a human agent now. "
        "They'll be in touch within 24 hours."
    ),
    "already_escalated": (
        "Your case is already with our review team. "
        "You'll hear from them within 24 hours — thank you for your patience."
    ),
    "system_error": (
        "I'm experiencing a technical issue right now and can't process your request automatically. "
        "I've escalated your case to our support team, who will follow up within 24 hours. "
        "Your case reference is {order_id}."
    ),
    "evidence_rejected": (
        "After reviewing the photo you provided, we need to pass your case to our "
        "specialist team for a closer look. They'll be in touch within 24 hours. "
        "Your case reference is {order_id}."
    ),
    "default": (
        "I've escalated your case to our support team. "
        "They'll follow up within 24 hours. "
        "Your case reference is {order_id}."
    ),
}


def generate_handoff_summary(
    state: dict,
    escalation_reason: str,
    escalation_type: str,
    llm_callable,
) -> tuple[str, str]:
    """
    Returns (agent_card_text, customer_message).

    agent_card_text: LLM-generated bullet list for the ops dashboard.
    customer_message: deterministic message for the customer.
    """
    order_id        = state.get("order_id", "N/A")
    customer_name   = state.get("customer_name", "Customer")
    complaint_type  = state.get("complaint_type", "unknown")
    claim_desc      = state.get("claim_description", "")
    risk_score      = state.get("risk_score", 0)
    risk_flags      = state.get("risk_flags", [])
    allergen_safety = state.get("allergen_safety", False)
    diet_restrict   = state.get("dietary_restriction")
    pending_offer   = state.get("pending_offer")
    offer_history   = state.get("offer_history", [])
    severity        = state.get("severity", "UNKNOWN")

    offer_summary = "No offer made yet."
    if pending_offer:
        offer_summary = (
            f"Pending offer: {pending_offer.get('resolution_type', 'unknown')}"
            f" — {pending_offer.get('offer_text', '')[:80]}"
        )
    elif offer_history:
        last = offer_history[-1]
        offer_summary = (
            f"Last offer: {last.get('offer', {}).get('resolution_type', 'unknown')}"
            f" — outcome: {last.get('outcome', 'unknown')}"
        )

    allergen_note = ""
    if allergen_safety:
        allergen_note = f"ALLERGEN SAFETY FLAG ACTIVE. Declared: {state.get('allergens_declared', [])}."
    elif diet_restrict:
        allergen_note = f"Dietary restriction declared: {diet_restrict}."

    open_issues = state.get("open_issues", [])
    open_issues_note = ""
    if open_issues:
        parts = [
            f"  - {oi.get('complaint_type', 'unknown')} (order {oi.get('order_id', 'N/A')})"
            for oi in open_issues
        ]
        open_issues_note = "Open unresolved issues:\n" + "\n".join(parts)

    evidence_note = ""
    ev = state.get("evidence_result")
    if ev:
        obs_text = "; ".join(ev.get("observations", [])[:3]) or "none"
        evidence_note = (
            f"Evidence verdict: {ev.get('verdict', 'unknown')} "
            f"(confidence: {ev.get('confidence', 0):.0%}) | "
            f"Duplicate: {ev.get('duplicate_image', False)} | "
            f"EXIF flag: {ev.get('exif_flag', False)}\n"
            f"Observations: {obs_text}\n"
            f"Thumbnail: {ev.get('thumbnail_path', 'N/A')}"
        )

    prompt = f"""
Order ID: {order_id}
Customer: {customer_name}
Complaint type: {complaint_type} | Severity: {severity}
Description: {claim_desc}
{allergen_note}
Risk score: {risk_score}/100  Flags: {', '.join(risk_flags) or 'none'}
{offer_summary}
{open_issues_note}
{evidence_note}
Escalation reason: {escalation_reason}
Escalation type: {escalation_type}
""".strip()

    # LLM generates the agent card
    try:
        response = llm_callable([
            SystemMessage(content=HANDOFF_SYSTEM),
            HumanMessage(content=prompt),
        ])
        agent_card = response.content
    except Exception as e:
        agent_card = f"[Handoff summary unavailable: {e}]\n\n{prompt}"

    # Deterministic customer message
    template = CUSTOMER_ESCALATION_MESSAGES.get(
        escalation_type,
        CUSTOMER_ESCALATION_MESSAGES["default"],
    )
    customer_msg = template.format(order_id=order_id, customer_name=customer_name)

    return agent_card, customer_msg


def build_escalation_record(
    state: dict,
    agent_card: str,
    escalation_reason: str,
    escalation_type: str,
) -> dict:
    """Build the structured dict that goes into escalations.jsonl and audit_log."""
    messages = state.get("messages", [])
    conversation = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
    ]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": state.get("session_id", ""),
        "order_id": state.get("order_id"),
        "customer_id": state.get("customer_id"),
        "customer_name": state.get("customer_name"),
        "complaint_type": state.get("complaint_type"),
        "severity": state.get("severity", "UNKNOWN"),
        "allergen_safety": state.get("allergen_safety", False),
        "escalation_reason": escalation_reason,
        "escalation_type": escalation_type,
        "risk_score": state.get("risk_score", 0),
        "risk_flags": state.get("risk_flags", []),
        "agent_card": agent_card,
        "offer_history": state.get("offer_history", []),
        "open_issues": state.get("open_issues", []),
        "conversation_history": conversation,
    }
