"""
Customer-facing tool functions — called by the LangGraph pipeline during live chat.

These are actions a customer can trigger through the chatbot:
  - Look up their order without knowing the order ID
  - Check live delivery status and delay
  - Cancel an in-progress order
  - Request compensation for a late delivery
  - Check the status of a support action they raised

Contrast with ops_tools.py, which is for the business/operations team.
"""
import json
from datetime import datetime, timezone
from typing import Optional

from app.tools import _conn, make_action_id, verify_order_belongs_to_customer


# ── Tool 1: get_active_order ──────────────────────────────────────────────────

def get_active_order(customer_id: str) -> dict:
    """
    Find the customer's most recent unresolved order without needing an order ID.
    Useful when a customer says "where's my order?" without providing a number.
    """
    with _conn() as conn:
        # Prefer an order that is still in-flight
        row = conn.execute("""
            SELECT * FROM orders
            WHERE customer_id = ?
              AND status NOT IN ('delivered', 'refund', 'partial_refund', 'cancelled')
            ORDER BY rowid DESC
            LIMIT 1
        """, (customer_id,)).fetchone()

        if row is None:
            # Fall back to most recent completed order
            row = conn.execute("""
                SELECT * FROM orders
                WHERE customer_id = ?
                ORDER BY rowid DESC
                LIMIT 1
            """, (customer_id,)).fetchone()

    if row is None:
        return {"found": False, "order": None}

    result = dict(row)
    if isinstance(result.get("items"), str):
        result["items"] = json.loads(result["items"])
    return {"found": True, "order": result}


# ── Tool 2: get_delivery_status ───────────────────────────────────────────────

def get_delivery_status(customer_id: str, order_id: str) -> dict:
    """
    Return live delivery status and delay for a specific order.
    Calculates minutes_past_promised_time from estimated vs actual delivery times.
    Verifies the order belongs to the customer before returning data.
    """
    with _conn() as conn:
        order = verify_order_belongs_to_customer(conn, order_id, customer_id)
        if order is None:
            return {
                "success": False,
                "error_code": "ORDER_NOT_FOUND",
                "message": f"Order {order_id} not found for this customer.",
            }

        delivery = conn.execute(
            "SELECT * FROM deliveries WHERE order_id = ?", (order_id,)
        ).fetchone()

    if delivery is None:
        return {
            "success": False,
            "error_code": "DELIVERY_DATA_UNAVAILABLE",
            "message": "Delivery information is not yet available for this order.",
        }

    est_min = order["estimated_delivery_min"] or 0
    act_min = order["actual_delivery_min"] or 0
    delay_minutes = max(0, act_min - est_min)

    return {
        "success": True,
        "order_id": order_id,
        "delivery_status": delivery["status"],
        "estimated_delivery_min": est_min,
        "actual_delivery_min": act_min if act_min else None,
        "delay_minutes": delay_minutes,
        "last_location": delivery["last_location"],
        "delivered_at": delivery["delivered_at"],
    }


# ── Tool 3: cancel_order ─────────────────────────────────────────────────────

def cancel_order(
    customer_id: str,
    order_id: str,
    reason: str,
    customer_confirmed: bool,
) -> dict:
    """
    Cancel an in-progress order on the customer's request.
    Guards: customer confirmation required, order must not already be delivered/cancelled,
    and duplicate cancellation is blocked via support_actions audit log.
    """
    if not customer_confirmed:
        return {
            "success": False,
            "error_code": "CUSTOMER_CONFIRMATION_REQUIRED",
            "message": "Please confirm you want to cancel this order before I proceed.",
        }

    conn = _conn()

    order = verify_order_belongs_to_customer(conn, order_id, customer_id)
    if order is None:
        conn.close()
        return {
            "success": False,
            "error_code": "ORDER_NOT_FOUND",
            "message": f"Order {order_id} not found for this customer.",
        }

    if order["status"] in ("delivered", "cancelled"):
        conn.close()
        return {
            "success": False,
            "error_code": "ORDER_CANNOT_BE_CANCELLED",
            "message": f"This order is already {order['status']} and cannot be cancelled.",
        }

    # Block duplicate cancellation
    existing = conn.execute("""
        SELECT action_id FROM support_actions
        WHERE order_id = ? AND action_type = 'CANCELLATION' AND status = 'COMPLETED'
    """, (order_id,)).fetchone()
    if existing:
        conn.close()
        return {
            "success": False,
            "error_code": "ALREADY_CANCELLED",
            "message": "A cancellation was already processed for this order.",
            "action_id": existing["action_id"],
        }

    action_id = make_action_id(conn)
    created_at = datetime.now(timezone.utc).isoformat()

    conn.execute("UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (order_id,))
    conn.execute("UPDATE deliveries SET status = 'CANCELLED' WHERE order_id = ?", (order_id,))
    conn.execute("""
        INSERT INTO support_actions
        (action_id, customer_id, order_id, action_type, amount, reason, status, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (action_id, customer_id, order_id, "CANCELLATION", order["total"], reason, "COMPLETED", created_at))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "action_id": action_id,
        "order_id": order_id,
        "action_type": "CANCELLATION",
        "status": "COMPLETED",
        "refund_amount": order["total"],
        "created_at": created_at,
    }


# ── Tool 4: apply_compensation ────────────────────────────────────────────────

def apply_compensation(
    customer_id: str,
    order_id: str,
    amount: float,
    reason: str,
    customer_confirmed: bool,
) -> dict:
    """
    Apply a compensation credit for a late delivery.
    Guards: confirmation required, delay must cross policy threshold,
    amount capped by policy, duplicate compensation blocked.
    Replaces the bare set_partial_refund_status for customer-initiated claims.
    """
    if not customer_confirmed:
        return {
            "success": False,
            "error_code": "CUSTOMER_CONFIRMATION_REQUIRED",
            "message": "Please confirm you want to accept the compensation offer before I apply it.",
        }

    conn = _conn()

    order = verify_order_belongs_to_customer(conn, order_id, customer_id)
    if order is None:
        conn.close()
        return {
            "success": False,
            "error_code": "ORDER_NOT_FOUND",
            "message": f"Order {order_id} not found for this customer.",
        }

    # Check for duplicate compensation
    existing = conn.execute("""
        SELECT action_id FROM support_actions
        WHERE order_id = ? AND action_type = 'COMPENSATION' AND status = 'COMPLETED'
    """, (order_id,)).fetchone()
    if existing:
        conn.close()
        return {
            "success": False,
            "error_code": "COMPENSATION_ALREADY_APPLIED",
            "message": "Compensation has already been applied for this order.",
            "action_id": existing["action_id"],
        }

    # Load policy for the issue type
    issue_type = "late_delivery" if "late" in reason.lower() else reason.split("_")[0] + "_" + reason.split("_")[1] if "_" in reason else reason
    policy = conn.execute(
        "SELECT * FROM policies WHERE issue_type = ?", (issue_type,)
    ).fetchone()

    if policy is None:
        policy = conn.execute(
            "SELECT * FROM policies WHERE issue_type = 'missing_item'"
        ).fetchone()

    # Cap the amount at policy maximum
    max_comp = policy["max_auto_compensation"] if policy else 100.0
    if amount <= 0:
        conn.close()
        return {
            "success": False,
            "error_code": "INVALID_AMOUNT",
            "message": "Compensation amount must be greater than zero.",
        }
    if amount > max_comp:
        conn.close()
        return {
            "success": False,
            "error_code": "AMOUNT_EXCEEDS_POLICY",
            "message": f"The requested amount exceeds our policy maximum of ${max_comp:.2f}.",
            "policy_max": max_comp,
        }

    action_id = make_action_id(conn)
    created_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "UPDATE orders SET status = 'partial_refund', refund_amount = ? WHERE order_id = ?",
        (amount, order_id),
    )
    conn.execute("""
        INSERT INTO support_actions
        (action_id, customer_id, order_id, action_type, amount, reason, status, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (action_id, customer_id, order_id, "COMPENSATION", amount, reason, "COMPLETED", created_at))
    conn.commit()
    conn.close()

    return {
        "success": True,
        "action_id": action_id,
        "order_id": order_id,
        "action_type": "COMPENSATION",
        "amount": amount,
        "status": "COMPLETED",
        "created_at": created_at,
    }


# ── Tool 5: get_action_status ─────────────────────────────────────────────────

def get_action_status(customer_id: str, action_id: str) -> dict:
    """
    Look up the result of a support action (compensation, cancellation, refund).
    Customers can call this to track what was done for their case.
    """
    with _conn() as conn:
        row = conn.execute("""
            SELECT action_id, order_id, action_type, amount, reason, status, created_at, notes
            FROM support_actions
            WHERE action_id = ? AND customer_id = ?
        """, (action_id, customer_id)).fetchone()

    if row is None:
        return {
            "success": False,
            "error_code": "ACTION_NOT_FOUND",
            "message": f"No support action {action_id} found for this customer.",
        }

    result = {
        "success": True,
        "action_id": row["action_id"],
        "order_id": row["order_id"],
        "action_type": row["action_type"],
        "reason": row["reason"],
        "status": row["status"],
        "created_at": row["created_at"],
    }
    if row["amount"] is not None:
        result["amount"] = row["amount"]
    if row["notes"]:
        result["notes"] = row["notes"]
    return result
