"""
Operations / business-side tool functions.

These are used by the internal team (support agents, managers) to:
  - Read structured policy rules for each issue type
  - View and resolve open escalations
  - See orders flagged for human review
  - Pull summary stats for the support dashboard

Not called directly by the customer-facing chatbot pipeline.
Contrast with customer_tools.py, which is what the AI uses during live chat.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import ESCALATIONS_LOG
from app.tools import _conn


# ── 1: get_support_policy ─────────────────────────────────────────────────────

def get_support_policy(issue_type: str) -> dict:
    """
    Return all severity levels for a given issue type.
    After the Phase-0 schema change, the policies table uses a
    (issue_type, severity) composite primary key.
    """
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM policies WHERE issue_type = ? ORDER BY severity",
            (issue_type,),
        ).fetchall()

    if not rows:
        return {
            "success": False,
            "error_code": "POLICY_NOT_FOUND",
            "message": f"No structured policy found for issue type '{issue_type}'.",
        }

    levels = {}
    for row in rows:
        try:
            allowed = json.loads(row["allowed_resolutions"])
        except Exception:
            allowed = []
        levels[row["severity"]] = {
            "allowed_resolutions":  allowed,
            "max_compensation_pct": row["max_compensation_pct"],
            "max_compensation_abs": row["max_compensation_abs"],
            "compensation_form":    row["compensation_form"],
            "auto_approve_limit":   row["auto_approve_limit"],
            "requires_confirm":     bool(row["requires_confirm"]),
            "policy_version":       row["policy_version"],
        }

    return {
        "success":    True,
        "issue_type": issue_type,
        "levels":     levels,
    }


# ── 2: get_pending_escalations ────────────────────────────────────────────────

def get_pending_escalations() -> list[dict]:
    """
    Return all orders currently flagged as 'humancheck' (escalated to ops team).
    Reads from the DB rather than the JSONL flat file so it can be filtered/sorted.
    Pairs each order with its escalation record from escalations.jsonl when present.
    """
    with _conn() as conn:
        rows = conn.execute("""
            SELECT order_id, customer_id, customer_name, email, total, status
            FROM orders
            WHERE status = 'humancheck'
            ORDER BY rowid DESC
        """).fetchall()

    # Index JSONL escalation records by order_id for enrichment
    escalation_map: dict = {}
    log_path = Path(ESCALATIONS_LOG)
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                oid = rec.get("order_id", "")
                escalation_map[oid] = rec
            except json.JSONDecodeError:
                pass

    result = []
    for r in rows:
        entry = dict(r)
        if entry["order_id"] in escalation_map:
            esc = escalation_map[entry["order_id"]]
            entry["escalation_reason"] = esc.get("escalation_reason")
            entry["risk_score"] = esc.get("risk_score")
            entry["risk_flags"] = esc.get("risk_flags", [])
            entry["escalated_at"] = esc.get("timestamp")
        result.append(entry)

    return result


# ── 3: get_pending_human_checks ───────────────────────────────────────────────

def get_pending_human_checks() -> list[dict]:
    """
    Same as get_pending_escalations but also includes orders where a
    support action was applied but the agent needs to verify outcome.
    Returns orders in 'humancheck' status plus any PENDING support actions.
    """
    with _conn() as conn:
        escalated = conn.execute("""
            SELECT o.order_id, o.customer_id, o.customer_name, o.total,
                   o.status AS order_status, 'ESCALATED' AS check_reason
            FROM orders o
            WHERE o.status = 'humancheck'
        """).fetchall()

        pending_actions = conn.execute("""
            SELECT o.order_id, o.customer_id, o.customer_name, o.total,
                   o.status AS order_status,
                   'PENDING_ACTION: ' || a.action_type AS check_reason
            FROM support_actions a
            JOIN orders o ON o.order_id = a.order_id
            WHERE a.status = 'PENDING'
        """).fetchall()

    return [dict(r) for r in escalated] + [dict(r) for r in pending_actions]


# ── 4: resolve_escalation ─────────────────────────────────────────────────────

def resolve_escalation(
    order_id: str,
    resolution: str,
    agent_notes: str,
    final_status: str = "resolved",
) -> dict:
    """
    Mark an escalated order as resolved by a human agent.
    resolution: 'full_refund' | 'partial_refund' | 'deny' | 'other'
    final_status: the order status to write (e.g. 'refund', 'delivered', 'closed')
    Writes an audit record to support_actions.
    """
    valid_resolutions = {
        "full_refund", "partial_refund", "coupon", "store_credit",
        "replacement", "priority_handling", "apology", "explanation",
        "deny", "other",
    }
    if resolution not in valid_resolutions:
        return {
            "success": False,
            "error_code": "INVALID_RESOLUTION",
            "message": f"resolution must be one of: {', '.join(sorted(valid_resolutions))}",
        }

    with _conn() as conn:
        order = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()

        if order is None:
            return {
                "success": False,
                "error_code": "ORDER_NOT_FOUND",
                "message": f"Order {order_id} not found.",
            }

        action_id_row = conn.execute("SELECT COUNT(*) AS c FROM support_actions").fetchone()
        action_id = f"ACT-{70000 + action_id_row['c'] + 1}"
        created_at = datetime.now(timezone.utc).isoformat()

        conn.execute(
            "UPDATE orders SET status = ? WHERE order_id = ?",
            (final_status, order_id),
        )
        conn.execute("""
            INSERT INTO support_actions
            (action_id, customer_id, order_id, action_type, reason, status, created_at, notes)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            action_id,
            order["customer_id"],
            order_id,
            "HUMAN_RESOLUTION",
            resolution,
            "COMPLETED",
            created_at,
            agent_notes,
        ))

    return {
        "success": True,
        "action_id": action_id,
        "order_id": order_id,
        "resolution": resolution,
        "final_status": final_status,
        "agent_notes": agent_notes,
        "resolved_at": created_at,
    }


# ── 5: get_support_stats ──────────────────────────────────────────────────────

def get_support_stats() -> dict:
    """
    Summary metrics for the ops dashboard:
    decision counts, total compensation paid, open escalations, cancellations.
    """
    with _conn() as conn:
        order_counts = conn.execute("""
            SELECT status, COUNT(*) AS cnt
            FROM orders
            GROUP BY status
        """).fetchall()

        action_totals = conn.execute("""
            SELECT action_type, COUNT(*) AS cnt, COALESCE(SUM(amount), 0) AS total
            FROM support_actions
            WHERE status = 'COMPLETED'
            GROUP BY action_type
        """).fetchall()

        open_escalations = conn.execute(
            "SELECT COUNT(*) AS cnt FROM orders WHERE status = 'humancheck'"
        ).fetchone()

    return {
        "order_status_breakdown": {r["status"]: r["cnt"] for r in order_counts},
        "completed_actions": {
            r["action_type"]: {"count": r["cnt"], "total_amount": r["total"]}
            for r in action_totals
        },
        "open_escalations": open_escalations["cnt"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
