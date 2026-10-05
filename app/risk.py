"""
Deterministic, auditable fraud risk scoring.
Returns a 0-100 score and a list of human-readable flags.
Based on the build plan's rule engine — no ML training needed.
"""
from dataclasses import dataclass, field


@dataclass
class RefundHistory:
    count_last_7_days: int = 0
    count_last_30_days: int = 0
    total_lifetime: int = 0
    same_issue_repeated: bool = False
    last_issues: list[str] = field(default_factory=list)


def compute_risk_score(
    history: RefundHistory,
    claim_amount: float,
    order_value: float,
) -> tuple[int, list[str]]:
    """Return (risk_score, flags). Score > 50 → escalate."""
    score = 0
    flags: list[str] = []

    # Frequent refunds in past week
    if history.count_last_7_days >= 3:
        score += 40
        flags.append(f"{history.count_last_7_days} refund requests in the last 7 days")
    elif history.count_last_7_days >= 2:
        score += 20
        flags.append(f"{history.count_last_7_days} refund requests in the last 7 days")

    # Claim nearly covers the whole order
    if order_value > 0 and claim_amount > order_value * 0.9:
        score += 30
        pct = int(claim_amount / order_value * 100)
        flags.append(f"Claim amount is {pct}% of order value (${claim_amount:.2f} of ${order_value:.2f})")

    # Same complaint type used before
    if history.same_issue_repeated:
        score += 20
        flags.append("Identical issue type claimed in a previous request")

    # High volume over 30 days
    if history.count_last_30_days >= 5:
        score += 15
        flags.append(f"{history.count_last_30_days} claims in the last 30 days")

    # Lifetime anomaly
    if history.total_lifetime >= 10:
        score += 10
        flags.append(f"High lifetime claim count: {history.total_lifetime} total claims")

    return min(score, 100), flags
