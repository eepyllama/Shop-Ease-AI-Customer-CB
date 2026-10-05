"""
One-time setup script: creates SQLite DB and seeds mock orders from data/orders.json.
Run once before starting the server: python setup_db.py
"""
import json
import sqlite3
from pathlib import Path

from app.config import DB_PATH

ORDERS_FILE = Path("data/orders.json")


def setup():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Create orders table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id        TEXT PRIMARY KEY,
            customer_id     TEXT NOT NULL,
            customer_name   TEXT,
            email           TEXT,
            items           TEXT,  -- JSON
            total           REAL,
            status          TEXT DEFAULT 'delivered',
            estimated_delivery_min INTEGER,
            actual_delivery_min    INTEGER,
            payment_method  TEXT,
            payment_status  TEXT,
            refund_amount   REAL DEFAULT 0
        )
    """)

    # Create refund history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS refund_history (
            customer_id         TEXT PRIMARY KEY,
            count_last_7_days   INTEGER DEFAULT 0,
            count_last_30_days  INTEGER DEFAULT 0,
            total_lifetime      INTEGER DEFAULT 0,
            last_issues         TEXT DEFAULT '[]'  -- JSON array
        )
    """)

    # Create sessions table (for persisted session state)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            state_json  TEXT,
            updated_at  TEXT
        )
    """)

    # ── NEW: Delivery tracking ────────────────────────────────────────────────
    # Separate from orders so delivery status can update independently.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS deliveries (
            order_id                TEXT PRIMARY KEY,
            status                  TEXT NOT NULL DEFAULT 'DELIVERED',
            estimated_arrival_min   INTEGER,
            actual_arrival_min      INTEGER,
            last_location           TEXT,
            delivered_at            TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        )
    """)

    # ── NEW: Support actions audit log ────────────────────────────────────────
    # Every compensation, cancellation, and refund is written here.
    # Prevents duplicate compensation on the same order.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS support_actions (
            action_id   TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            order_id    TEXT NOT NULL,
            action_type TEXT NOT NULL,  -- COMPENSATION | CANCELLATION | FULL_REFUND
            amount      REAL,
            reason      TEXT NOT NULL,
            status      TEXT NOT NULL,  -- COMPLETED | REJECTED | PENDING
            created_at  TEXT NOT NULL,
            notes       TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        )
    """)

    # ── NEW: Structured policy rules ──────────────────────────────────────────
    # Machine-readable complement to the RAG policy.md.
    # Adjudication reads these for exact thresholds.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            issue_type                  TEXT PRIMARY KEY,
            minor_delay_threshold_min   INTEGER NOT NULL DEFAULT 15,
            compensation_threshold_min  INTEGER NOT NULL DEFAULT 30,
            full_refund_threshold_min   INTEGER NOT NULL DEFAULT 90,
            max_auto_compensation       REAL    NOT NULL DEFAULT 100.0,
            customer_confirmation_req   INTEGER NOT NULL DEFAULT 1,
            cancellation_allowed        INTEGER NOT NULL DEFAULT 1,
            policy_version              TEXT    NOT NULL DEFAULT '2026-10-01'
        )
    """)

    conn.commit()

    # ── Seed policies ─────────────────────────────────────────────────────────
    policies = [
        # issue_type, minor_min, comp_min, full_min, max_comp, confirm_req, cancel_ok
        ("late_delivery",  15, 30, 90,  100.0, 1, 1),
        ("missing_item",    0,  0,  0,  100.0, 0, 0),
        ("quality_issue",   0,  0,  0,  100.0, 0, 0),
        ("wrong_order",     0,  0,  0,  100.0, 0, 0),
        ("payment_issue",   0,  0,  0,  100.0, 1, 0),
    ]
    for p in policies:
        cur.execute("""
            INSERT OR IGNORE INTO policies
            (issue_type, minor_delay_threshold_min, compensation_threshold_min,
             full_refund_threshold_min, max_auto_compensation,
             customer_confirmation_req, cancellation_allowed)
            VALUES (?,?,?,?,?,?,?)
        """, p)
    conn.commit()

    # Seed from mock data
    if not ORDERS_FILE.exists():
        print(f"Warning: {ORDERS_FILE} not found. Skipping seed.")
        conn.close()
        return

    data = json.loads(ORDERS_FILE.read_text())

    for o in data.get("orders", []):
        cur.execute("""
            INSERT OR REPLACE INTO orders
            (order_id, customer_id, customer_name, email, items, total, status,
             estimated_delivery_min, actual_delivery_min, payment_method, payment_status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            o["order_id"],
            o["customer_id"],
            o["customer_name"],
            o["email"],
            json.dumps(o["items"]),
            o["total"],
            o.get("status", "delivered"),
            o.get("estimated_delivery_min"),
            o.get("actual_delivery_min"),
            o.get("payment_method"),
            o.get("payment_status"),
        ))

    for cust_id, h in data.get("refund_histories", {}).items():
        cur.execute("""
            INSERT OR REPLACE INTO refund_history
            (customer_id, count_last_7_days, count_last_30_days, total_lifetime, last_issues)
            VALUES (?,?,?,?,?)
        """, (
            cust_id,
            h.get("count_last_7_days", 0),
            h.get("count_last_30_days", 0),
            h.get("total_lifetime", 0),
            json.dumps(h.get("last_issues", [])),
        ))

    # ── Seed deliveries from orders ───────────────────────────────────────────
    rows = cur.execute("SELECT order_id, status, estimated_delivery_min, actual_delivery_min FROM orders").fetchall()
    for r in rows:
        oid, status, est_min, act_min = r
        delivery_status = "DELIVERED" if status in ("delivered", "refund", "partial_refund") else "OUT_FOR_DELIVERY"
        cur.execute("""
            INSERT OR IGNORE INTO deliveries
            (order_id, status, estimated_arrival_min, actual_arrival_min)
            VALUES (?,?,?,?)
        """, (oid, delivery_status, est_min, act_min))

    conn.commit()
    conn.close()
    print(f"✓ Database created at '{DB_PATH}' with {len(data.get('orders', []))} orders.")


if __name__ == "__main__":
    setup()
    print("✓ Setup complete. Run: uvicorn app.main:app --reload")
