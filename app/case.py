"""
Case lifecycle management.

A Case is the core business entity for the platform — one per customer
complaint attempt.  Cases are stored in the `cases` DB table and used by
the business dashboard and audit trail.

The LangGraph session continues to hold conversation state.
Cases hold the structured summary: status, risk, resolution, timestamps.

Lifecycle (see Universal Resolution Lifecycle in spec):
  new → active → resolved | escalated | insufficient_evidence | action_failed

Case IDs are formatted as CASE-<timestamp-ms>-<4-hex-random>.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from app.tools import _conn

logger = logging.getLogger(__name__)


# ── Case status values ────────────────────────────────────────────────────────
class CaseStatus:
    NEW                   = "new"
    ACTIVE                = "active"
    WAITING_CONFIRMATION  = "waiting_confirmation"
    EXECUTING             = "executing"
    RESOLVED              = "resolved"
    ESCALATED             = "escalated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ACTION_FAILED         = "action_failed"


def _generate_case_id() -> str:
    import random
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    suffix = format(random.randint(0, 0xFFFF), "04x")
    return f"CASE-{ts}-{suffix}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Create ────────────────────────────────────────────────────────────────────

def create_case(
    session_id: str,
    tenant_id: str,
    customer_id: Optional[str] = None,
    reference_id: Optional[str] = None,
    issue_type: Optional[str] = None,
    claim_text: Optional[str] = None,
) -> str:
    """
    Create a new case record.  Returns the case_id.
    Called on the first turn of a new session.
    """
    case_id = _generate_case_id()
    now = _now()
    try:
        with _conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO cases
                (case_id, tenant_id, session_id, customer_id, reference_id,
                 issue_type, claim_text, status, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                case_id, tenant_id, session_id, customer_id, reference_id,
                issue_type, claim_text, CaseStatus.NEW, now, now,
            ))
    except Exception as exc:
        logger.warning("case.create_case: DB write failed: %s", exc)
    return case_id


def get_case_for_session(session_id: str) -> Optional[dict]:
    """Return the case linked to a session, or None."""
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        logger.warning("case.get_case_for_session: %s", exc)
        return None


def update_case(
    case_id: str,
    *,
    status: Optional[str] = None,
    customer_id: Optional[str] = None,
    reference_id: Optional[str] = None,
    issue_type: Optional[str] = None,
    claim_text: Optional[str] = None,
    risk_score: Optional[int] = None,
    evidence_confidence: Optional[int] = None,
    recommended_resolution: Optional[str] = None,
    actual_resolution: Optional[str] = None,
    escalation_status: Optional[str] = None,
) -> None:
    """Update mutable fields on a case.  Only non-None kwargs are written."""
    fields: list[tuple] = []
    if status                is not None: fields.append(("status",                 status))
    if customer_id           is not None: fields.append(("customer_id",            customer_id))
    if reference_id          is not None: fields.append(("reference_id",           reference_id))
    if issue_type            is not None: fields.append(("issue_type",             issue_type))
    if claim_text            is not None: fields.append(("claim_text",             claim_text))
    if risk_score            is not None: fields.append(("risk_score",             risk_score))
    if evidence_confidence   is not None: fields.append(("evidence_confidence",    evidence_confidence))
    if recommended_resolution is not None: fields.append(("recommended_resolution", recommended_resolution))
    if actual_resolution     is not None: fields.append(("actual_resolution",      actual_resolution))
    if escalation_status     is not None: fields.append(("escalation_status",      escalation_status))

    if not fields:
        return

    fields.append(("updated_at", _now()))
    set_clause = ", ".join(f"{col} = ?" for col, _ in fields)
    values = [v for _, v in fields] + [case_id]

    try:
        with _conn() as conn:
            conn.execute(
                f"UPDATE cases SET {set_clause} WHERE case_id = ?",
                values,
            )
    except Exception as exc:
        logger.warning("case.update_case(%s): %s", case_id, exc)


# ── Dashboard queries ─────────────────────────────────────────────────────────

def list_cases(
    tenant_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Return cases for the dashboard, newest first."""
    conditions: list[str] = []
    params: list = []
    if tenant_id:
        conditions.append("tenant_id = ?")
        params.append(tenant_id)
    if status:
        conditions.append("status = ?")
        params.append(status)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params += [limit, offset]

    try:
        with _conn() as conn:
            rows = conn.execute(f"""
                SELECT * FROM cases
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("case.list_cases: %s", exc)
        return []


def get_case(case_id: str) -> Optional[dict]:
    """Return a single case by ID."""
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        logger.warning("case.get_case(%s): %s", case_id, exc)
        return None


def case_stats(tenant_id: Optional[str] = None) -> dict:
    """Aggregate counts for the dashboard overview."""
    cond = "WHERE tenant_id = ?" if tenant_id else ""
    params = [tenant_id] if tenant_id else []
    try:
        with _conn() as conn:
            rows = conn.execute(f"""
                SELECT status, COUNT(*) AS cnt
                FROM cases {cond}
                GROUP BY status
            """, params).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}
        total = sum(counts.values())
        resolved = counts.get("resolved", 0)
        escalated = counts.get("escalated", 0)
        return {
            "total":    total,
            "active":   counts.get("active", 0) + counts.get("waiting_confirmation", 0),
            "resolved": resolved,
            "escalated": escalated,
            "auto_resolved_pct": round(resolved / total * 100, 1) if total else 0,
            "escalated_pct":     round(escalated / total * 100, 1) if total else 0,
            "by_status": counts,
        }
    except Exception as exc:
        logger.warning("case.case_stats: %s", exc)
        return {"total": 0, "active": 0, "resolved": 0, "escalated": 0}
