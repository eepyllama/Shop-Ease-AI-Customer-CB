"""
One-time setup and seeding script:
- Creates SQLite tables (multi-tenant, cases, evidence, audit, confirmations, orders).
- Seeds tenants (Demo FoodCo and FashionCo), configs, and structured policies in INR.
- Seeds orders, refund histories, and deliveries from data/orders.json.
- Seeds realistic demonstration cases for both tenants with complete evidence,
  policy evaluations, confirmations, and audit timelines.
Run: python setup_db.py
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import DB_PATH
from app.schema import ensure_schema

ORDERS_FILE = Path("data/orders.json")


def setup():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # 1. Base legacy tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id        TEXT PRIMARY KEY,
            tenant_id       TEXT NOT NULL DEFAULT 'demo_foodco',
            customer_id     TEXT NOT NULL,
            customer_name   TEXT,
            email           TEXT,
            items           TEXT,
            total           REAL,
            status          TEXT DEFAULT 'delivered',
            estimated_delivery_min INTEGER,
            actual_delivery_min    INTEGER,
            payment_method  TEXT,
            payment_status  TEXT,
            refund_amount   REAL DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS refund_history (
            customer_id         TEXT PRIMARY KEY,
            tenant_id           TEXT NOT NULL DEFAULT 'demo_foodco',
            count_last_7_days   INTEGER DEFAULT 0,
            count_last_30_days  INTEGER DEFAULT 0,
            total_lifetime      INTEGER DEFAULT 0,
            last_issues         TEXT DEFAULT '[]'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS deliveries (
            order_id                TEXT PRIMARY KEY,
            tenant_id               TEXT NOT NULL DEFAULT 'demo_foodco',
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
            tenant_id   TEXT NOT NULL DEFAULT 'demo_foodco',
            customer_id TEXT NOT NULL,
            order_id    TEXT NOT NULL,
            case_id     TEXT,
            confirmation_id TEXT,
            action_type TEXT NOT NULL,
            amount      REAL,
            reason      TEXT NOT NULL,
            status      TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            notes       TEXT,
            verification_result TEXT,
            verified_at TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            issue_type                  TEXT PRIMARY KEY,
            minor_delay_threshold_min   INTEGER NOT NULL DEFAULT 15,
            compensation_threshold_min  INTEGER NOT NULL DEFAULT 30,
            full_refund_threshold_min   INTEGER NOT NULL DEFAULT 90,
            max_auto_compensation       REAL    NOT NULL DEFAULT 1500.0,
            customer_confirmation_req   INTEGER NOT NULL DEFAULT 1,
            cancellation_allowed        INTEGER NOT NULL DEFAULT 1,
            policy_version              TEXT    NOT NULL DEFAULT '2026-10-01'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            state_json  TEXT,
            updated_at  TEXT
        )
    """)
    conn.commit()

    # 2. Ensure all platform tables
    ensure_schema(conn)

    # 3. Seed Tenants
    tenants = [
        ("demo_foodco", "Demo FoodCo", "food_delivery", "active"),
        ("fashionco", "FashionCo", "fashion", "active"),
    ]
    for t in tenants:
        conn.execute(
            "INSERT OR REPLACE INTO tenants (tenant_id, name, industry, status) VALUES (?,?,?,?)",
            t,
        )

    # 4. Seed Tenant Configs
    food_config = {
        "brand_name": "Demo FoodCo Support",
        "primary_color": "#EA580C",
        "logo": "🍕",
        "welcome_message": "Hi! Welcome to Demo FoodCo support. How can we help resolve an issue with your order?",
        "currency": "INR",
        "reference_label": "Order ID (e.g. ORD-002)",
        "issue_types": [
            {"id": "missing_item", "label": "Missing Item", "requires_evidence": False},
            {"id": "quality_issue", "label": "Quality / Food Condition", "requires_evidence": True},
            {"id": "wrong_order", "label": "Wrong Order Received", "requires_evidence": True},
            {"id": "late_delivery", "label": "Late Delivery", "requires_evidence": False},
            {"id": "payment_issue", "label": "Payment / Billing Issue", "requires_evidence": False},
        ],
        "media_upload": {"enabled": True, "max_mb": 10, "accept": "image/*,video/*"},
        "support_hours": "Mon–Sun, 8:00–23:00 IST",
        "footer_text": "Powered by ResolveIQ",
        "risk_thresholds": {"escalate_above": 50, "review_above": 30},
        "escalation": {"channel": "ops_queue", "sla_hours": 24},
        "integration_type": "mock",
    }

    fashion_config = {
        "brand_name": "FashionCo Concierge",
        "primary_color": "#4F46E5",
        "logo": "👗",
        "welcome_message": "Hello! Welcome to FashionCo. How can we assist with your order or delivery today?",
        "currency": "INR",
        "reference_label": "Order Number (e.g. ORD-FAS-101)",
        "issue_types": [
            {"id": "damaged_item", "label": "Damaged / Defective Garment", "requires_evidence": True},
            {"id": "wrong_item", "label": "Wrong Item or Size Received", "requires_evidence": True},
            {"id": "missing_item", "label": "Missing Item from Package", "requires_evidence": False},
            {"id": "late_delivery", "label": "Delayed Shipment", "requires_evidence": False},
        ],
        "media_upload": {"enabled": True, "max_mb": 15, "accept": "image/*,video/*"},
        "support_hours": "Mon–Sat, 9:00–21:00 IST",
        "footer_text": "Powered by ResolveIQ",
        "risk_thresholds": {"escalate_above": 50, "review_above": 30},
        "escalation": {"channel": "fashion_care", "sla_hours": 12},
        "integration_type": "mock",
    }

    conn.execute(
        "INSERT OR REPLACE INTO tenant_configs (tenant_id, config_json) VALUES (?,?)",
        ("demo_foodco", json.dumps(food_config)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO tenant_configs (tenant_id, config_json) VALUES (?,?)",
        ("fashionco", json.dumps(fashion_config)),
    )

    # 5. Seed Structured Tenant Policies
    food_policies = [
        ("missing_item", {
            "resolution": "ITEM_REFUND",
            "max_auto_amount": 1000,
            "human_review_above": 1000,
            "requires_evidence": False,
            "requires_confirmation": True,
            "report_window_hours": 24,
            "contradiction_outcome": "escalate",
        }),
        ("quality_issue", {
            "resolution": "SEVERITY_PCT",
            "max_auto_amount": 1200,
            "human_review_above": 1200,
            "requires_evidence": True,
            "requires_confirmation": True,
            "report_window_hours": 2,
            "severity_pcts": {"minor": 0.3, "moderate": 0.5, "severe": 1.0},
        }),
        ("wrong_order", {
            "resolution": "FULL_REFUND",
            "max_auto_amount": 1500,
            "human_review_above": 1500,
            "requires_evidence": False,
            "requires_confirmation": True,
            "report_window_hours": 24,
        }),
        ("late_delivery", {
            "resolution": "LATE_DELIVERY",
            "max_auto_amount": 1500,
            "human_review_above": 1500,
            "requires_confirmation": True,
            "delay_bands": [
                {"over_min": 90, "type": "FULL_REFUND"},
                {"over_min": 60, "type": "PERCENT_VOUCHER", "pct": 20},
                {"over_min": 30, "type": "FLAT_VOUCHER", "amount": 100},
            ],
        }),
        ("payment_issue", {
            "resolution": "STANDARD_REFUND",
            "max_auto_amount": 1000,
            "human_review_above": 1000,
            "requires_confirmation": True,
            "auto_resolution": "ESCALATE",
        }),
    ]

    for it, rules in food_policies:
        conn.execute(
            "INSERT OR REPLACE INTO tenant_policies (tenant_id, issue_type, rules_json, policy_version) "
            "VALUES (?,?,?,?)",
            ("demo_foodco", it, json.dumps(rules), "v1"),
        )

    fashion_policies = [
        ("damaged_item", {
            "resolution": "ITEM_REFUND",
            "max_auto_amount": 2500,
            "human_review_above": 2500,
            "requires_evidence": True,
            "requires_confirmation": True,
            "report_window_hours": 168,
            "contradiction_outcome": "escalate",
        }),
        ("wrong_item", {
            "resolution": "ITEM_REFUND",
            "max_auto_amount": 2500,
            "human_review_above": 2500,
            "requires_evidence": True,
            "requires_confirmation": True,
            "report_window_hours": 168,
        }),
        ("missing_item", {
            "resolution": "ITEM_REFUND",
            "max_auto_amount": 2000,
            "human_review_above": 2000,
            "requires_evidence": False,
            "requires_confirmation": True,
            "report_window_hours": 48,
            "contradiction_outcome": "escalate",
        }),
        ("late_delivery", {
            "resolution": "LATE_DELIVERY",
            "max_auto_amount": 500,
            "human_review_above": 500,
            "requires_confirmation": True,
            "delay_bands": [
                {"over_min": 2880, "type": "FLAT_VOUCHER", "amount": 200},
            ],
        }),
    ]

    for it, rules in fashion_policies:
        conn.execute(
            "INSERT OR REPLACE INTO tenant_policies (tenant_id, issue_type, rules_json, policy_version) "
            "VALUES (?,?,?,?)",
            ("fashionco", it, json.dumps(rules), "v1"),
        )

    # 6. Legacy Policies
    legacy_policies = [
        ("late_delivery", 15, 30, 90, 1500.0, 1, 1),
        ("missing_item", 0, 0, 0, 1000.0, 1, 0),
        ("quality_issue", 0, 0, 0, 1200.0, 1, 0),
        ("wrong_order", 0, 0, 0, 1500.0, 1, 0),
        ("payment_issue", 0, 0, 0, 1000.0, 1, 0),
    ]
    for p in legacy_policies:
        conn.execute("""
            INSERT OR REPLACE INTO policies
            (issue_type, minor_delay_threshold_min, compensation_threshold_min,
             full_refund_threshold_min, max_auto_compensation,
             customer_confirmation_req, cancellation_allowed)
            VALUES (?,?,?,?,?,?,?)
        """, p)

    # 7. Seed Orders
    if ORDERS_FILE.exists():
        data = json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
        for o in data.get("orders", []):
            tid = o.get("tenant_id", "demo_foodco")
            conn.execute("""
                INSERT OR REPLACE INTO orders
                (order_id, tenant_id, customer_id, customer_name, email, items, total, status,
                 estimated_delivery_min, actual_delivery_min, payment_method, payment_status, refund_amount)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                o["order_id"],
                tid,
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
                o.get("refund_amount", 0),
            ))

        for cust_id, h in data.get("refund_histories", {}).items():
            tid = "fashionco" if "FAS" in cust_id else "demo_foodco"
            conn.execute("""
                INSERT OR REPLACE INTO refund_history
                (customer_id, tenant_id, count_last_7_days, count_last_30_days, total_lifetime, last_issues)
                VALUES (?,?,?,?,?,?)
            """, (
                cust_id,
                tid,
                h.get("count_last_7_days", 0),
                h.get("count_last_30_days", 0),
                h.get("total_lifetime", 0),
                json.dumps(h.get("last_issues", [])),
            ))

    # 8. Seed Deliveries
    rows = conn.execute("SELECT order_id, tenant_id, status, estimated_delivery_min, actual_delivery_min FROM orders").fetchall()
    for r in rows:
        oid, tid, status, est_min, act_min = r["order_id"], r["tenant_id"], r["status"], r["estimated_delivery_min"], r["actual_delivery_min"]
        delivery_status = "DELIVERED" if status in ("delivered", "refund", "partial_refund") else "OUT_FOR_DELIVERY"
        conn.execute("""
            INSERT OR REPLACE INTO deliveries
            (order_id, tenant_id, status, estimated_arrival_min, actual_arrival_min)
            VALUES (?,?,?,?,?)
        """, (oid, tid, delivery_status, est_min, act_min))

    # 9. Seed Realistic Demonstration Cases for Dashboard
    demo_cases = [
        # Food Tenant: Resolved missing item
        ("CASE-1001", "demo_foodco", "CUST-001", "Ananya Sharma", "missing_item",
         "Sweet lassi was missing from the order delivery bag.", "RESOLVED", "ORD-001", "order",
         "INR", 10, "LOW", 90, "ITEM_REFUND: 80 INR", "apply_compensation: ₹80", None),
        # Food Tenant: Waiting confirmation
        ("CASE-1002", "demo_foodco", "CUST-002", "Vikram Malhotra", "missing_item",
         "Garlic bread missing from family combo.", "WAITING_FOR_CUSTOMER_CONFIRMATION", "ORD-002", "order",
         "INR", 15, "LOW", 85, "ITEM_REFUND: 89 INR", None, None),
        # Food Tenant: Escalated high-risk customer
        ("CASE-1003", "demo_foodco", "CUST-007", "Priya Sundaram", "missing_item",
         "Claims missing chicken tikka masala.", "ESCALATED", "ORD-007", "order",
         "INR", 70, "HIGH_RISK_REVIEW", 40, "ESCALATE", None, "PENDING_HUMAN_REVIEW"),
        # Food Tenant: Contradiction claim rejected
        ("CASE-1004", "demo_foodco", "CUST-008", "Harsh Joshi", "missing_item",
         "Fries were missing from order.", "REJECTED", "ORD-008", "order",
         "INR", 20, "LOW", 20, "REJECT", None, None),
        # Fashion Tenant: Damaged shirt resolved with photo evidence
        ("CASE-2001", "fashionco", "CUST-FAS-001", "Priya Sharma", "damaged_item",
         "Classic Cotton Oxford Shirt arrived with tear on right sleeve.", "RESOLVED", "ORD-FAS-101", "order",
         "INR", 10, "LOW", 94, "ITEM_REFUND: 1499 INR", "apply_compensation: ₹1499", None),
        # Fashion Tenant: Wrong item waiting confirmation
        ("CASE-2002", "fashionco", "CUST-FAS-002", "Rohan Verma", "wrong_item",
         "Received black hoodie instead of Vintage Denim Jacket.", "WAITING_FOR_CUSTOMER_CONFIRMATION", "ORD-FAS-102", "order",
         "INR", 15, "LOW", 88, "ITEM_REFUND: 2899 INR", None, None),
        # Fashion Tenant: Luxury order above limit escalated
        ("CASE-2003", "fashionco", "CUST-FAS-003", "Neha Gupta", "damaged_item",
         "Silk dress seam unstitched.", "ESCALATED", "ORD-FAS-103", "order",
         "INR", 10, "LOW", 92, "ESCALATE", None, "PENDING_HUMAN_REVIEW"),
    ]

    for c in demo_cases:
        conn.execute("""
            INSERT OR REPLACE INTO cases
            (case_id, tenant_id, customer_id, customer_name, issue_type, claim_text,
             status, reference_id, reference_type, currency, risk_score, risk_level,
             evidence_confidence, recommended_resolution, actual_resolution, escalation_status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, c)

    # Seed Support Actions for the resolved cases
    conn.execute("""
        INSERT OR REPLACE INTO support_actions
        (action_id, tenant_id, customer_id, order_id, case_id, confirmation_id,
         action_type, amount, reason, status, created_at, notes, verification_result, verified_at)
        VALUES
        ('ACT-70001', 'demo_foodco', 'CUST-001', 'ORD-001', 'CASE-1001', 'CONF-FOOD-01',
         'COMPENSATION', 80, 'missing_item', 'COMPLETED', datetime('now', '-2 hours'),
         'Missing sweet lassi refund', 'VERIFIED', datetime('now', '-2 hours')),
        ('ACT-70002', 'fashionco', 'CUST-FAS-001', 'ORD-FAS-101', 'CASE-2001', 'CONF-FAS-01',
         'COMPENSATION', 1499, 'damaged_item', 'COMPLETED', datetime('now', '-1 hours'),
         'Damaged Oxford shirt refund verified with photo evidence', 'VERIFIED', datetime('now', '-1 hours'))
    """)

    # Seed Audit Events for CASE-1001 and CASE-2001
    conn.execute("""
        INSERT INTO audit_events (tenant_id, case_id, event_type, actor, metadata_json, created_at)
        VALUES
        ('demo_foodco', 'CASE-1001', 'CASE_CREATED', 'CUSTOMER', '{"issue_type": "missing_item"}', datetime('now', '-2 hours', '-5 minutes')),
        ('demo_foodco', 'CASE-1001', 'ORDER_VERIFIED', 'SYSTEM', '{"order_id": "ORD-001", "total": 550}', datetime('now', '-2 hours', '-4 minutes')),
        ('demo_foodco', 'CASE-1001', 'RESOLUTION_RECOMMENDED', 'SYSTEM', '{"amount": 80, "currency": "INR"}', datetime('now', '-2 hours', '-3 minutes')),
        ('demo_foodco', 'CASE-1001', 'CONFIRMATION_ACCEPTED', 'CUSTOMER', '{"amount": 80}', datetime('now', '-2 hours', '-2 minutes')),
        ('demo_foodco', 'CASE-1001', 'GATEWAY_ACTION_EXECUTED', 'SYSTEM', '{"tool": "apply_compensation", "amount": 80}', datetime('now', '-2 hours', '-1 minutes')),
        ('demo_foodco', 'CASE-1001', 'ACTION_VERIFIED', 'SYSTEM', '{"action_id": "ACT-70001", "read_back": "VERIFIED"}', datetime('now', '-2 hours')),
        ('fashionco', 'CASE-2001', 'CASE_CREATED', 'CUSTOMER', '{"issue_type": "damaged_item"}', datetime('now', '-1 hours', '-10 minutes')),
        ('fashionco', 'CASE-2001', 'EVIDENCE_UPLOADED', 'CUSTOMER', '{"filename": "torn_shirt.jpg", "size_bytes": 184200}', datetime('now', '-1 hours', '-8 minutes')),
        ('fashionco', 'CASE-2001', 'EVIDENCE_VISUAL_ANALYZED', 'SYSTEM', '{"damage_detected": true, "confidence": 0.94}', datetime('now', '-1 hours', '-7 minutes')),
        ('fashionco', 'CASE-2001', 'RESOLUTION_RECOMMENDED', 'SYSTEM', '{"amount": 1499, "currency": "INR"}', datetime('now', '-1 hours', '-5 minutes')),
        ('fashionco', 'CASE-2001', 'CONFIRMATION_ACCEPTED', 'CUSTOMER', '{"amount": 1499}', datetime('now', '-1 hours', '-3 minutes')),
        ('fashionco', 'CASE-2001', 'GATEWAY_ACTION_EXECUTED', 'SYSTEM', '{"tool": "apply_compensation", "amount": 1499}', datetime('now', '-1 hours', '-1 minutes')),
        ('fashionco', 'CASE-2001', 'ACTION_VERIFIED', 'SYSTEM', '{"action_id": "ACT-70002", "read_back": "VERIFIED"}', datetime('now', '-1 hours'))
    """)

    conn.commit()
    conn.close()
    print(f"[OK] Database setup complete at '{DB_PATH}' with multi-tenant seed data and sample cases.")


if __name__ == "__main__":
    setup()
