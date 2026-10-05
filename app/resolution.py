"""
Resolution engine — the single authority on what the AI may offer a customer.

The LLM must NEVER compute compensation amounts or policy confidence.
This module does both deterministically:
  - compute_allowed_options()  → what resolution types are allowed for (issue_type, severity)
  - compute_compensation()     → the exact dollar amount, as a function of order_value only
  - validate_resolution()      → guards against LLM picking outside the allowed list
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from app.config import MAX_AUTO_APPROVE_ABS, ALLERGEN_ALWAYS_ESCALATE
from app.tools import _conn


@dataclass
class AllowedOptions:
    allowed: list[str]          # ordered cheapest → costliest
    max_pct: float              # max compensation as % of order value (0–100)
    max_abs: float              # absolute ceiling in local currency
    form: str                   # "coupon" | "credit" | "cash" | "none"
    auto_approve_limit: float   # 0 = always requires human approval
    requires_confirm: bool      # customer must confirm before execution


_FALLBACK_OPTIONS = AllowedOptions(
    allowed=["apology", "escalate"],
    max_pct=0.0,
    max_abs=0.0,
    form="none",
    auto_approve_limit=0.0,
    requires_confirm=False,
)


def compute_allowed_options(issue_type: str, severity: str) -> AllowedOptions:
    """
    Query the policies table for (issue_type, severity) and return AllowedOptions.
    Falls back to ["apology", "escalate"] when no policy row exists.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM policies WHERE issue_type = ? AND severity = ?",
            (issue_type, severity),
        ).fetchone()

    if row is None:
        # Try with just issue_type + MEDIUM as a safer fallback
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM policies WHERE issue_type = ? AND severity = 'MEDIUM'",
                (issue_type,),
            ).fetchone()

    if row is None:
        return _FALLBACK_OPTIONS

    try:
        allowed = json.loads(row["allowed_resolutions"])
    except (json.JSONDecodeError, TypeError):
        allowed = ["apology", "escalate"]

    # HIGH / CRITICAL complaints deserve real remediation — remove token options
    if severity in ("HIGH", "CRITICAL"):
        allowed = [o for o in allowed if o not in ("apology", "explanation")]
        if not allowed:
            allowed = ["full_refund", "escalate"]

    return AllowedOptions(
        allowed=allowed,
        max_pct=float(row["max_compensation_pct"]),
        max_abs=float(row["max_compensation_abs"]),
        form=str(row["compensation_form"]),
        auto_approve_limit=float(row["auto_approve_limit"]),
        requires_confirm=bool(row["requires_confirm"]),
    )


def compute_compensation(
    order_value: float,
    options: AllowedOptions,
    resolution_type: str,
) -> Optional[float]:
    """
    Compute the exact compensation amount deterministically.
    Returns None for resolution types that carry no monetary value
    (apology, explanation, replacement, priority_handling, escalate).
    """
    non_monetary = {"apology", "explanation", "replacement", "priority_handling", "escalate"}
    if resolution_type in non_monetary:
        return None

    if options.max_pct <= 0 and options.max_abs <= 0:
        return None

    pct_amount = order_value * (options.max_pct / 100.0)
    amount = min(pct_amount, options.max_abs) if options.max_abs > 0 else pct_amount

    # Apply the global absolute auto-approve cap
    if amount > MAX_AUTO_APPROVE_ABS:
        # Still return the amount but callers will force human approval
        pass

    return round(amount, 2)


def validate_resolution(
    resolution_type: str,
    options: AllowedOptions,
    allergen_safety: bool = False,
) -> str:
    """
    Ensure the LLM's chosen resolution_type is in the allowed list.
    If not (hallucination / out-of-bounds choice), fall back to allowed[0].

    allergen_safety=True forces the result to include "escalate" — if the allowed
    list contains it, we use "full_refund" (the costliest non-escalate option),
    and always pair it with escalation in the caller.
    """
    if allergen_safety and ALLERGEN_ALWAYS_ESCALATE:
        # Must escalate regardless of what the LLM picked
        return "full_refund"  # caller also sets escalated=True

    if resolution_type in options.allowed:
        return resolution_type

    # Invalid choice — use cheapest allowed option
    return options.allowed[0] if options.allowed else "apology"


def requires_human_approval(amount: Optional[float], options: AllowedOptions) -> bool:
    """
    True when the computed amount exceeds the policy's auto_approve_limit
    OR the global MAX_AUTO_APPROVE_ABS cap.
    A limit of 0 always requires human approval.
    """
    if options.auto_approve_limit == 0:
        return True
    if amount is None:
        return False
    if amount > options.auto_approve_limit:
        return True
    if amount > MAX_AUTO_APPROVE_ABS:
        return True
    return False
