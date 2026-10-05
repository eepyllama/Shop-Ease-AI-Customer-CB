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

    # ── Core tables ───────────────────────────────────────────────────────────

    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id                TEXT PRIMARY KEY,
            customer_id             TEXT NOT NULL,
            customer_name           TEXT,
            email                   TEXT,
            items                   TEXT,  -- JSON
            total                   REAL,
            status                  TEXT DEFAULT 'delivered',
            estimated_delivery_min  INTEGER,
            actual_delivery_min     INTEGER,
            payment_method          TEXT,
            payment_status          TEXT,
            refund_amount           REAL DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS refund_history (
            customer_id         TEXT PRIMARY KEY,
            count_last_7_days   INTEGER DEFAULT 0,
            count_last_30_days  INTEGER DEFAULT 0,
            total_lifetime      INTEGER DEFAULT 0,
            last_issues         TEXT DEFAULT '[]'  -- JSON array
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            state_json  TEXT,
            updated_at  TEXT
        )
    """)

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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS support_actions (
            action_id   TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            order_id    TEXT NOT NULL,
            action_type TEXT NOT NULL,
            amount      REAL,
            reason      TEXT NOT NULL,
            status      TEXT NOT NULL,  -- COMPLETED | REJECTED | PENDING
            created_at  TEXT NOT NULL,
            notes       TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        )
    """)

    # ── Audit log — every pipeline event is written here ─────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id      TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            order_id    TEXT,
            customer_id TEXT,
            event_type  TEXT NOT NULL,
            payload     TEXT,  -- JSON
            created_at  TEXT NOT NULL
        )
    """)

    # ── Policies table — (issue_type, severity) composite PK ─────────────────
    # Drop and recreate so the composite-key schema is applied cleanly.
    cur.execute("DROP TABLE IF EXISTS policies")
    cur.execute("""
        CREATE TABLE policies (
            issue_type              TEXT NOT NULL,
            severity                TEXT NOT NULL,  -- LOW | MEDIUM | HIGH | CRITICAL
            allowed_resolutions     TEXT NOT NULL DEFAULT '[]',  -- JSON array, cheapest→costliest
            max_compensation_pct    REAL NOT NULL DEFAULT 0.0,   -- % of order value
            max_compensation_abs    REAL NOT NULL DEFAULT 0.0,   -- absolute ceiling
            compensation_form       TEXT NOT NULL DEFAULT 'none',  -- coupon|credit|cash|none
            auto_approve_limit      REAL NOT NULL DEFAULT 0.0,   -- 0 = always needs human
            requires_confirm        INTEGER NOT NULL DEFAULT 1,
            policy_version          TEXT NOT NULL DEFAULT '2026-10-01',
            PRIMARY KEY (issue_type, severity)
        )
    """)

    conn.commit()

    # ── Seed policies (issue_type, severity, allowed_resolutions,
    #                   max_pct, max_abs, form, auto_approve, confirm) ──────────
    policies = [
        # late_delivery
        ("late_delivery", "LOW",      '["apology","explanation"]',              0,   0,   "none",   0,   0),
        ("late_delivery", "MEDIUM",   '["coupon","store_credit"]',              10,  50,  "coupon", 50,  1),
        ("late_delivery", "HIGH",     '["partial_refund","full_refund"]',       50,  100, "cash",   100, 1),
        ("late_delivery", "CRITICAL", '["full_refund","escalate"]',             100, 500, "cash",   0,   1),
        # missing_item
        ("missing_item",  "LOW",      '["apology","coupon"]',                   10,  20,  "coupon", 20,  1),
        ("missing_item",  "MEDIUM",   '["partial_refund"]',                     50,  50,  "cash",   50,  1),
        ("missing_item",  "HIGH",     '["partial_refund","full_refund"]',       100, 100, "cash",   100, 1),
        ("missing_item",  "CRITICAL", '["full_refund","escalate"]',             100, 500, "cash",   0,   1),
        # quality_issue
        ("quality_issue", "LOW",      '["apology","coupon"]',                   20,  30,  "coupon", 30,  1),
        ("quality_issue", "MEDIUM",   '["partial_refund"]',                     50,  100, "cash",   100, 1),
        ("quality_issue", "HIGH",     '["full_refund","escalate"]',             100, 100, "cash",   0,   1),
        ("quality_issue", "CRITICAL", '["full_refund","escalate"]',             100, 500, "cash",   0,   1),
        # wrong_order
        ("wrong_order",   "LOW",      '["replacement","partial_refund"]',       50,  100, "cash",   100, 1),
        ("wrong_order",   "MEDIUM",   '["replacement","full_refund"]',          100, 100, "cash",   100, 1),
        ("wrong_order",   "HIGH",     '["full_refund","escalate"]',             100, 500, "cash",   0,   1),
        ("wrong_order",   "CRITICAL", '["full_refund","escalate"]',             100, 500, "cash",   0,   1),
        # dietary_mismatch (veg/non-veg, religious)
        ("dietary_mismatch", "HIGH",     '["replacement","full_refund"]',       100, 100, "cash",   100, 1),
        ("dietary_mismatch", "CRITICAL", '["full_refund","escalate"]',          100, 500, "cash",   0,   1),
        # allergen_violation — always CRITICAL, auto_approve=0 means human required
        ("allergen_violation", "CRITICAL", '["full_refund","escalate"]',        100, 500, "cash",   0,   0),
        # freebie_missing
        ("freebie_missing", "LOW",    '["apology","coupon"]',                   5,   10,  "coupon", 10,  1),
        ("freebie_missing", "MEDIUM", '["coupon","store_credit"]',              10,  20,  "coupon", 20,  1),
        # payment_issue
        ("payment_issue", "LOW",      '["explanation","apology"]',              0,   0,   "none",   0,   0),
        ("payment_issue", "MEDIUM",   '["store_credit"]',                       100, 100, "credit", 100, 1),
        ("payment_issue", "HIGH",     '["full_refund","escalate"]',             100, 500, "cash",   0,   1),
    ]

    for p in policies:
        cur.execute("""
            INSERT OR IGNORE INTO policies
            (issue_type, severity, allowed_resolutions,
             max_compensation_pct, max_compensation_abs, compensation_form,
             auto_approve_limit, requires_confirm)
            VALUES (?,?,?,?,?,?,?,?)
        """, p)

    conn.commit()

    # ── Seed from mock data ───────────────────────────────────────────────────

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

    # ── Seed deliveries ───────────────────────────────────────────────────────
    rows = cur.execute(
        "SELECT order_id, status, estimated_delivery_min, actual_delivery_min FROM orders"
    ).fetchall()

    status_map = {
        "delivered":      "DELIVERED",
        "refund":         "DELIVERED",
        "partial_refund": "DELIVERED",
        "cancelled":      "CANCELLED",
        "out_for_delivery": "OUT_FOR_DELIVERY",
        "preparing":      "PREPARING",
        "humancheck":     "DELIVERED",
    }
    for r in rows:
        oid, status, est_min, act_min = r
        delivery_status = status_map.get(status, "DELIVERED")
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
