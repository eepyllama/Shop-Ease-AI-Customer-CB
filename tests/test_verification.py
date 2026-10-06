"""
Verification layer tests — 6 scenarios from the spec.

All scenarios are unit-level (no LLM calls, no graph invocation).
Vision calls are mocked via VerificationResult.from_evidence().
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.verification import VerificationResult, get_verifier_type, verify_timestamps
from app.severity import classify_severity
from app.graph import _lower_severity


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _load_order(order_id: str) -> dict:
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "orders.json")
    with open(data_path) as f:
        data = json.load(f)
    return next(o for o in data["orders"] if o["order_id"] == order_id)


# ── Scenario 1: late claim matching data → correct tier ───────────────────────

def test_late_claim_matching_data_verified():
    """ORD-005 (95-min late): claim says '2 hours late' → verified at HIGH."""
    order = _load_order("ORD-005")
    result = verify_timestamps(order, "my order was 2 hours late")

    assert result.status == "verified"
    assert result.method == "timestamps"
    assert any("data_severity:HIGH" in f for f in result.flags)
    assert result.confidence >= 0.9


# ── Scenario 2: claim contradicts data ───────────────────────────────────────

def test_claim_contradicts_data():
    """ORD-013 (8-min delay, below threshold): claim '2 hours late' → contradicted."""
    order = _load_order("ORD-013")
    result = verify_timestamps(order, "my order was 2 hours late")

    assert result.status == "contradicted"
    assert "claim_contradicts_data" in result.flags
    assert result.confidence >= 0.85
    assert any("Delay: 8 min" in obs for obs in result.observations)


# ── Scenario 3: insect + "I ate some" → CRITICAL escalation ─────────────────

def test_foreign_object_critical_ingested():
    """foreign_object complaint with ingestion keywords → CRITICAL, allergen_safety=True."""
    severity, allergen_safety = classify_severity(
        complaint_type="foreign_object",
        claim_description="I found an insect in my food and I ate some of it",
        items=[{"name": "Pasta", "veg": True, "allergens": []}],
        dietary_restriction=None,
        allergens_declared=[],
    )
    assert severity == "CRITICAL"
    assert allergen_safety is True


def test_foreign_object_high_no_ingestion():
    """foreign_object without ingestion keywords → HIGH, allergen_safety=False."""
    severity, allergen_safety = classify_severity(
        complaint_type="foreign_object",
        claim_description="I found a hair in my burger",
        items=[{"name": "Burger", "veg": False, "allergens": ["gluten"]}],
        dietary_restriction=None,
        allergens_declared=[],
    )
    assert severity == "HIGH"
    assert allergen_safety is False


# ── Scenario 4: duplicate photo → escalation ─────────────────────────────────

def test_duplicate_photo_contradicted():
    """Duplicate image detected → VerificationResult status=contradicted, flag=duplicate_image."""
    evidence = {
        "verdict": "consistent",
        "duplicate_image": True,
        "confidence": 0.95,
        "observations": ["Image hash matches a prior claim from this account."],
    }
    result = VerificationResult.from_evidence(evidence)

    assert result.status == "contradicted"
    assert "duplicate_image" in result.flags
    assert result.method == "photo"


def test_inconsistent_photo_contradicted():
    """Vision verdict=inconsistent → VerificationResult status=contradicted, flag=vision_inconsistent."""
    evidence = {
        "verdict": "inconsistent",
        "duplicate_image": False,
        "confidence": 0.80,
        "observations": ["Food in photo does not match items in the order."],
    }
    result = VerificationResult.from_evidence(evidence)

    assert result.status == "contradicted"
    assert "vision_inconsistent" in result.flags


# ── Scenario 5: upload skipped → severity tier lowered ───────────────────────

def test_skip_gives_inconclusive():
    """Skipping the photo upload → from_evidence with empty dict → inconclusive."""
    result = VerificationResult.from_evidence({})

    assert result.status == "inconclusive"
    assert result.method == "photo"


def test_severity_lowered_one_tier():
    """_lower_severity drops exactly one tier within HIGH/MEDIUM/LOW, bottoming at LOW.
    CRITICAL is not in the tier list (it escalates directly) so it falls back to LOW."""
    assert _lower_severity("HIGH")   == "MEDIUM"
    assert _lower_severity("MEDIUM") == "LOW"
    assert _lower_severity("LOW")    == "LOW"


# ── Scenario 6: in-progress order → not_applicable ───────────────────────────

def test_in_progress_order_not_applicable():
    """ORD-011 (preparing, actual_delivered_at=null) → not_applicable + order_in_progress flag."""
    order = _load_order("ORD-011")
    result = verify_timestamps(order, "my order is taking too long")

    assert result.status == "not_applicable"
    assert "order_in_progress" in result.flags


def test_out_for_delivery_not_applicable():
    """ORD-012 (out_for_delivery) → not_applicable."""
    order = _load_order("ORD-012")
    result = verify_timestamps(order, "still waiting for my order")

    assert result.status == "not_applicable"
    assert "order_in_progress" in result.flags


# ── Verifier registry ─────────────────────────────────────────────────────────

def test_verifier_type_registry():
    assert get_verifier_type("wrong_order")     == "photo"
    assert get_verifier_type("foreign_object")  == "photo"
    assert get_verifier_type("late_delivery")   == "timestamps"
    assert get_verifier_type("payment_issue")   == "none"
    assert get_verifier_type("freebie_missing") == "none"
    assert get_verifier_type("refund_status")   == "none"
    assert get_verifier_type("other")           == "none"
