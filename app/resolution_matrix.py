"""
Resolution matrix — the authoritative source for which options are shown per
(complaint_type × severity).  No if-chains; just dict lookup + two helpers.

Lookup order: exact (complaint_type, severity) key first, then ("*", severity)
default.  Returns a flat list of resolution-type strings ordered from most
thorough to least (what the customer sees as buttons).

CRITICAL is handled upstream (allergen path) — this module returns [] for it,
which the caller treats as immediate escalation.
"""
from __future__ import annotations

# ── Option list keyed by (complaint_type, severity) ──────────────────────────
# HIGH: "escalate" included upfront (customers see all paths immediately)
# MEDIUM / LOW: "escalate" added only after first typed dissatisfaction
MATRIX: dict[tuple[str, str], list[str]] = {
    # HIGH defaults
    ("*",             "HIGH"):   ["replacement", "full_refund", "escalate"],
    # HIGH overrides — delivery/payment never include replacement
    ("late_delivery", "HIGH"):   ["full_refund", "coupon", "escalate"],
    ("payment_issue", "HIGH"):   ["full_refund", "coupon", "escalate"],

    # MEDIUM defaults
    ("*",             "MEDIUM"): ["replacement", "partial_refund"],
    # MEDIUM overrides
    ("late_delivery", "MEDIUM"): ["coupon", "partial_refund"],
    ("payment_issue", "MEDIUM"): ["coupon", "partial_refund"],

    # LOW defaults
    ("*",             "LOW"):    ["coupon", "store_credit"],
    # CRITICAL: no options → caller escalates immediately
}

# These complaint types never include replacement in any severity
_NO_REPLACEMENT: frozenset[str] = frozenset({"late_delivery", "payment_issue"})


def get_options(complaint_type: str, severity: str) -> list[str]:
    """
    Return the initial option list for (complaint_type, severity).
    Empty list means immediate escalation (CRITICAL or no policy found).
    """
    if severity == "CRITICAL":
        return []

    options: list[str] = list(
        MATRIX.get((complaint_type, severity))
        or MATRIX.get(("*", severity))
        or []
    )

    # Guard: never offer replacement for delivery / payment complaints
    if complaint_type in _NO_REPLACEMENT and "replacement" in options:
        options = [o for o in options if o != "replacement"]

    return options


def get_options_after_dissatisfaction(complaint_type: str, severity: str) -> list[str]:
    """
    Options shown after the first typed dissatisfaction.
    MEDIUM/LOW gain "escalate" (Talk to human) if not already present.
    """
    options = get_options(complaint_type, severity)
    if severity in ("MEDIUM", "LOW") and "escalate" not in options:
        options = options + ["escalate"]
    return options


def compute_option_amount(
    option_type: str,
    order_total: float,
    complaint_type: str,
    severity: str,
) -> float | None:
    """
    Return the monetary compensation for one option, or None if non-monetary.
    full_refund is always 100 % of order_total.
    Other monetary types use the DB policy max for (complaint_type, severity).
    """
    non_monetary = {"replacement", "priority_handling", "apology", "explanation", "escalate"}
    if option_type in non_monetary:
        return None

    if option_type == "full_refund":
        return round(order_total, 2)

    # partial_refund, coupon, store_credit — defer to DB policy
    try:
        from app.resolution import compute_allowed_options, compute_compensation
        opts = compute_allowed_options(complaint_type, severity)
        return compute_compensation(order_total, opts, option_type)
    except Exception:
        # Fallback percentages when DB is unreachable
        _FALLBACK_PCT = {"partial_refund": 50.0, "coupon": 20.0, "store_credit": 25.0}
        pct = _FALLBACK_PCT.get(option_type, 0.0)
        return round(order_total * pct / 100, 2) if pct else None
