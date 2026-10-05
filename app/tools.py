"""
Tool functions — adapted from ComplaintBot's setup_db.py + setup_email.py.
Uses SQLite instead of PostgreSQL so the prototype runs with zero infra setup.
"""
import json
import sqlite3
import smtplib
import random
import string
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

from app.config import DB_PATH, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ESCALATIONS_LOG
from app.risk import RefundHistory


# ── database helpers ──────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def make_action_id(conn) -> str:
    row = conn.execute("SELECT COUNT(*) AS c FROM support_actions").fetchone()
    return f"ACT-{70000 + row['c'] + 1}"


def verify_order_belongs_to_customer(conn, order_id: str, customer_id: str):
    return conn.execute(
        "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
        (order_id, customer_id),
    ).fetchone()


def get_order_details(order_id: str) -> dict | None:
    """Return full order record, or None if not found."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    # items is stored as JSON
    if isinstance(result.get("items"), str):
        result["items"] = json.loads(result["items"])
    return result


def get_refund_history(customer_id: str) -> RefundHistory:
    """Return refund history object for risk scoring."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM refund_history WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    if not row:
        return RefundHistory()
    d = dict(row)
    last_issues = json.loads(d.get("last_issues", "[]"))
    return RefundHistory(
        count_last_7_days=d.get("count_last_7_days", 0),
        count_last_30_days=d.get("count_last_30_days", 0),
        total_lifetime=d.get("total_lifetime", 0),
        same_issue_repeated=False,  # set later in adjudication
        last_issues=last_issues,
    )


def set_refund_status(order_id: str) -> bool:
    """Mark order as refunded. Returns True on success."""
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE orders SET status = 'refund' WHERE order_id = ?", (order_id,)
            )
        return True
    except Exception:
        return False


def set_humancheck_status(order_id: str) -> bool:
    """Flag order for human review. Returns True on success."""
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE orders SET status = 'humancheck' WHERE order_id = ?", (order_id,)
            )
        return True
    except Exception:
        return False


def set_partial_refund_status(order_id: str, amount: float) -> bool:
    """Record a partial refund amount and mark accordingly."""
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE orders SET status = 'partial_refund', refund_amount = ? WHERE order_id = ?",
                (amount, order_id),
            )
        return True
    except Exception:
        return False


# ── voucher / email (from ComplaintBot's setup_email.py) ─────────────────────

def _generate_voucher_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def send_voucher_email(email: str, customer_name: str, voucher_value: str, reason: str) -> bool:
    """Send a voucher code by email. Returns True on success."""
    if not SMTP_USER or not SMTP_PASS:
        # Email not configured — log and return True so the flow continues
        print(f"[EMAIL SKIPPED] Would send voucher to {email}: {voucher_value} for '{reason}'")
        return True

    code = _generate_voucher_code()
    body = f"""Hi {customer_name},

We're sorry about your recent experience. As a goodwill gesture, here is your voucher:

  Code: {code}
  Value: {voucher_value}
  Reason: {reason}
  Valid for: 30 days

Thank you for your patience.

— Customer Support Team
"""
    msg = MIMEText(body)
    msg["Subject"] = f"Your voucher: {code}"
    msg["From"] = SMTP_USER
    msg["To"] = email

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, email, msg.as_string())
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


# ── escalation logging ────────────────────────────────────────────────────────

def log_escalation(
    session_id: str,
    order_id: str,
    reason: str,
    conversation: list[dict],
    order_details: dict,
    risk_score: int,
    risk_flags: list[str],
    suggested_action: str = "Manual review required",
) -> None:
    """Append a structured escalation record to escalations.jsonl."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "order_id": order_id,
        "escalation_reason": reason,
        "risk_score": risk_score,
        "risk_flags": risk_flags,
        "suggested_action": suggested_action,
        "order_details": order_details,
        "conversation_history": conversation,
    }
    with open(ESCALATIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── session store (in-memory for prototype) ───────────────────────────────────

_sessions: dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    return _sessions.get(session_id, {
        "session_id": session_id,
        "messages": [],
        "order_id": None,
        "phase": "greeting",
    })


def save_session(session_id: str, state: dict) -> None:
    _sessions[session_id] = state


def list_sessions() -> list[dict]:
    return [
        {"session_id": k, "phase": v.get("phase"), "order_id": v.get("order_id")}
        for k, v in _sessions.items()
    ]
