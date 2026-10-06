"""
Server-side confirmation tokens (Phase 2).

When the resolution engine proposes an action (refund, coupon, credit),
it creates a CONF-xxx token bound to the exact case + action + amount.
The customer's "Accept" button must present the conf_id back to the server,
which validates it before executing — the LLM never confirms on their behalf.

Lifecycle:  pending → accepted | rejected | expired
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.tools import _conn

logger = logging.getLogger(__name__)

_TTL_MINUTES = 15


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _gen_conf_id() -> str:
    return "CONF-" + uuid.uuid4().hex[:12].upper()


# ── Create ────────────────────────────────────────────────────────────────────

def create_confirmation(
    case_id: str,
    session_id: str,
    action_type: str,
    amount: float = 0.0,
    currency: str = "INR",
    ttl_minutes: int = _TTL_MINUTES,
) -> str:
    """
    Mint a new CONF-xxx token for a proposed action.
    Returns the conf_id.  Any previous pending token for this session is
    superseded (marked expired) before the new one is created.
    """
    conf_id = _gen_conf_id()
    now = _now()
    expires_at = (now + timedelta(minutes=ttl_minutes)).isoformat()

    try:
        with _conn() as conn:
            # Expire any outstanding pending tokens for this session
            conn.execute(
                "UPDATE confirmations SET status='expired', updated_at=? "
                "WHERE session_id=? AND status='pending'",
                (now.isoformat(), session_id),
            )
            conn.execute(
                """
                INSERT INTO confirmations
                  (conf_id, case_id, session_id, action_type, amount, currency,
                   status, expires_at, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (conf_id, case_id, session_id, action_type, amount, currency,
                 "pending", expires_at, now.isoformat(), now.isoformat()),
            )
    except Exception as exc:
        logger.warning("confirmation.create: DB write failed: %s", exc)

    return conf_id


# ── Validate ──────────────────────────────────────────────────────────────────

def validate_and_accept(
    conf_id: str,
    session_id: str,
    expected_action_type: Optional[str] = None,
) -> dict:
    """
    Validate a confirmation token for acceptance.

    Returns a dict with:
      ok: bool
      action_type: str
      amount: float
      error: str (only when ok=False)

    Side effect on success: marks the token as 'accepted'.
    """
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM confirmations WHERE conf_id=?", (conf_id,)
            ).fetchone()

            if not row:
                return {"ok": False, "error": "unknown confirmation token"}

            row = dict(row)

            if row["session_id"] != session_id:
                return {"ok": False, "error": "token belongs to a different session"}

            if row["status"] != "pending":
                return {"ok": False, "error": f"token already {row['status']}"}

            now = _now()
            expires_at = datetime.fromisoformat(row["expires_at"])
            if now > expires_at:
                conn.execute(
                    "UPDATE confirmations SET status='expired', updated_at=? WHERE conf_id=?",
                    (now.isoformat(), conf_id),
                )
                return {"ok": False, "error": "confirmation token expired"}

            if expected_action_type and row["action_type"] != expected_action_type:
                return {
                    "ok": False,
                    "error": f"token is for '{row['action_type']}', not '{expected_action_type}'",
                }

            conn.execute(
                "UPDATE confirmations SET status='accepted', updated_at=? WHERE conf_id=?",
                (now.isoformat(), conf_id),
            )
            return {
                "ok": True,
                "action_type": row["action_type"],
                "amount": row["amount"],
                "currency": row["currency"],
            }

    except Exception as exc:
        logger.warning("confirmation.validate: %s", exc)
        return {"ok": False, "error": str(exc)}


def reject_confirmation(conf_id: str) -> None:
    """Mark a pending token as rejected (customer declined)."""
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE confirmations SET status='rejected', updated_at=? WHERE conf_id=? AND status='pending'",
                (_now().isoformat(), conf_id),
            )
    except Exception as exc:
        logger.warning("confirmation.reject: %s", exc)


def get_confirmation(conf_id: str) -> Optional[dict]:
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM confirmations WHERE conf_id=?", (conf_id,)
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None
