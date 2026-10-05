"""
Escalation pre-check — runs as the first node of every turn.

Five triggers checked in priority order:
  1. session.escalated already True  → hard stop (acknowledge only)
  2. Allergen safety flag             → hard stop (advise + full_refund + escalate)
  3. Explicit human request in message → hard stop
  4. Near-identical to previous bot response (loop detection) → hard stop
  5. resolution_attempts >= MAX  OR  rejection_count >= MAX → soft escalation
     (these become a regular escalation after the pipeline runs once more)

Hard stop: bot acknowledges only; no further resolution attempt this session.
Soft stop: the current turn escalates — pipeline sets escalated=True, logs, and ends.
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from app.config import (
    MAX_RESOLUTION_ATTEMPTS,
    MAX_REJECTION_ATTEMPTS,
    REPEAT_RESPONSE_THRESHOLD,
    ALLERGEN_ALWAYS_ESCALATE,
)

_HUMAN_REQUEST_PATTERNS = [
    r"\bhuman\b", r"\bagent\b", r"\brepresentative\b", r"\bperson\b",
    r"\bspeak to someone\b", r"\btalk to someone\b", r"\breal person\b",
    r"\blive (agent|support|chat)\b", r"\bescalate\b",
    r"\bmanager\b", r"\bsupervisor\b",
]
_HUMAN_RE = re.compile("|".join(_HUMAN_REQUEST_PATTERNS), re.IGNORECASE)


def _token_overlap(a: str, b: str) -> float:
    """Simple bag-of-words Jaccard similarity."""
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _last_bot_response(messages: list) -> Optional[str]:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content
    return None


def _latest_human_message(messages: list) -> Optional[str]:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return None


def check_escalation(state: dict) -> dict:
    """
    Evaluate all escalation triggers and return a dict with:
      {
        "should_escalate": bool,
        "is_hard_stop":    bool,   # True = do not run any more pipeline nodes
        "reason":          str,
        "escalation_type": str,    # "already_escalated" | "allergen" | "human_request"
                                   # | "loop_detected" | "max_attempts" | "max_rejections"
      }
    """
    messages        = state.get("messages", [])
    escalated       = state.get("escalated", False)
    allergen_safety = state.get("allergen_safety", False)
    resolution_attempts = state.get("resolution_attempts", 0)
    offer_history   = state.get("offer_history", [])
    rejection_count = sum(1 for o in offer_history if o.get("outcome") == "rejected")

    user_text  = _latest_human_message(messages) or ""
    prev_bot   = _last_bot_response(messages[:-1]) if len(messages) > 1 else None

    # ── 1. Already escalated ─────────────────────────────────────────────────
    if escalated:
        return {
            "should_escalate": True,
            "is_hard_stop": True,
            "reason": "Session is already escalated to human review.",
            "escalation_type": "already_escalated",
        }

    # ── 2. Allergen safety ───────────────────────────────────────────────────
    if allergen_safety and ALLERGEN_ALWAYS_ESCALATE:
        return {
            "should_escalate": True,
            "is_hard_stop": True,
            "reason": "Allergen safety flag: customer declared allergen present in order.",
            "escalation_type": "allergen",
        }

    # ── 3. Explicit human request ────────────────────────────────────────────
    if _HUMAN_RE.search(user_text):
        return {
            "should_escalate": True,
            "is_hard_stop": True,
            "reason": "Customer explicitly requested a human agent.",
            "escalation_type": "human_request",
        }

    # ── 4. Loop detection (near-identical previous bot response) ─────────────
    if prev_bot:
        similarity = _token_overlap(user_text, prev_bot)
        if similarity >= REPEAT_RESPONSE_THRESHOLD:
            return {
                "should_escalate": True,
                "is_hard_stop": True,
                "reason": "Loop detected: outgoing response near-identical to previous turn.",
                "escalation_type": "loop_detected",
            }

    # ── 5. Max resolution attempts ───────────────────────────────────────────
    if resolution_attempts >= MAX_RESOLUTION_ATTEMPTS:
        return {
            "should_escalate": True,
            "is_hard_stop": False,
            "reason": f"Maximum resolution attempts reached ({resolution_attempts}/{MAX_RESOLUTION_ATTEMPTS}).",
            "escalation_type": "max_attempts",
        }

    # ── 6. Max offer rejections ──────────────────────────────────────────────
    if rejection_count >= MAX_REJECTION_ATTEMPTS:
        return {
            "should_escalate": True,
            "is_hard_stop": False,
            "reason": f"Customer rejected {rejection_count} resolution offers.",
            "escalation_type": "max_rejections",
        }

    return {
        "should_escalate": False,
        "is_hard_stop": False,
        "reason": "",
        "escalation_type": "",
    }
