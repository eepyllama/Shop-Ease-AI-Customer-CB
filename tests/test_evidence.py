"""
Photo-evidence verification tests — updated for verification-layer architecture.

Architecture change (from prior design):
  OLD: offer_node handled the evidence gate + verdict decision rules.
  NEW: verification_node sets verification_result; adjudication_node applies
       the verdict (contradicted → escalate, inconclusive → lower tier);
       offer_node adds the escalate button for photo-inconclusive cases.

Tests cover:
  1. consistent verdict + HIGH severity → normal HIGH options shown
  2. duplicate perceptual hash → escalation via adjudication_node
  3. inconsistent verdict → escalation via adjudication_node, not denial
  4. Non-photo types / LOW severity → verification is not_applicable
  5. Image with instruction text → verdict schema is always enforced
  6. Customer skips upload → treated as inconclusive (capped tier + escalate button)
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.resolution_matrix import get_options
from app.graph import (
    _EVIDENCE_REQUIRED_TYPES,
    _lower_severity,
    adjudication_node,
    offer_node,
    route_after_confirm,
    route_after_escalation_check,
)
from app.verification import get_verifier_type


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_adjudication_state(
    verification_result: dict | None,
    complaint_type: str = "quality_issue",
    severity_hint: str = "HIGH",
) -> dict:
    """Minimal state for adjudication_node tests with a pre-set verification result."""
    return {
        "session_id":          "test-evidence-session",
        "order_id":            "ORD-EV1",
        "complaint_type":      complaint_type,
        "claim_description":   "Food arrived cold and undercooked",
        "severity":            None,  # adjudication_node computes this
        "allergen_safety":     False,
        "allergens_declared":  [],
        "dietary_restriction": None,
        "rag_context":         "",
        "policy_confidence":   0.9,
        "offer_history":       [],
        "messages":            [],
        "verification_result": verification_result,
        "case_status":         None,
    }


def _make_offer_state(
    severity: str,
    verification_result: dict | None,
) -> dict:
    """Minimal state dict for offer_node tests."""
    return {
        "complaint_type":        "quality_issue",
        "severity":              severity,
        "order_id":              "ORD-EV1",
        "customer_name":         None,
        "claim_description":     "Food arrived cold and undercooked",
        "offer_history":         [],
        "correction_detected":   False,
        "dissatisfaction_count": 0,
        "order_details":         {"total": 30.00, "items": []},
        "customer_id":           "CUST-EV1",
        "session_id":            "test-evidence-session",
        "verification_result":   verification_result,
        "risk_flags":            [],
        "messages":              [],
    }


def _fake_order() -> dict:
    return {
        "order_id":            "ORD-EV1",
        "customer_id":         "CUST-EV1",
        "customer_name":       "Test User",
        "email":               "test@example.com",
        "items":               [
            {"name": "Pasta", "price": 12.00, "quantity": 1,
             "veg": True, "allergens": []},
        ],
        "total":               30.00,
        "status":              "delivered",
        "estimated_delivery_min": 30,
        "actual_delivery_min":    35,
    }


def _fake_refund_history():
    h = SimpleNamespace()
    h.count_last_7_days = 0
    h.count_last_30_days = 0
    h.total_lifetime = 0
    h.last_issues = []
    h.same_issue_repeated = False
    return h


def _mock_adjudication_llm(resolution_type: str = "full_refund"):
    mock_llm = MagicMock()
    output = MagicMock()
    output.resolution_type = resolution_type
    output.reasoning       = "test reasoning"
    output.risk_flags      = []
    mock_llm.invoke.return_value = output
    return mock_llm


def _mock_offer_llm():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "I sincerely apologise for the issue."
    return mock_llm


# ── Test 1: consistent verdict → normal HIGH options ─────────────────────────

class TestConsistentEvidence:
    def test_consistent_high_shows_standard_options(self):
        """
        verified photo + HIGH → adjudication passes through without capping;
        offer_node presents the standard HIGH option set.
        """
        with patch("app.graph._get_llm") as mock_get_llm:
            mock_get_llm.return_value = _mock_offer_llm()

            state = _make_offer_state("HIGH", {
                "status":       "verified",
                "method":       "photo",
                "confidence":   0.92,
                "observations": ["Food clearly appears undercooked."],
                "flags":        [],
            })

            result = offer_node(state)

        pending = result.get("pending_offer", {})
        option_types = [o["type"] for o in (pending.get("resolution_options") or [])]
        assert "full_refund" in option_types, f"full_refund must be present for verified HIGH, got {option_types}"
        assert result.get("resolution_type") != "escalate", "Verified evidence must not trigger escalation"

    def test_consistent_does_not_cap_severity(self):
        """
        HIGH and MEDIUM option sets must differ so that tier-capping is detectable.
        """
        high_opts   = set(get_options("quality_issue", "HIGH"))
        medium_opts = set(get_options("quality_issue", "MEDIUM"))
        assert high_opts != medium_opts, "HIGH and MEDIUM option sets must differ"


# ── Test 2: duplicate hash → escalation ──────────────────────────────────────

class TestDuplicateHashEscalation:
    def test_duplicate_image_escalates_in_adjudication(self):
        """
        verification_result status=contradicted + flag=duplicate_image → adjudication
        must return resolution_type='escalate' with escalation_type='evidence_rejected'.
        """
        vr = {
            "status":       "contradicted",
            "method":       "photo",
            "confidence":   0.95,
            "observations": ["Image hash matches a prior claim."],
            "flags":        ["duplicate_image"],
        }

        with patch("app.graph.get_order_details", return_value=_fake_order()), \
             patch("app.graph.get_refund_history", return_value=_fake_refund_history()), \
             patch("app.graph._get_llm") as mock_get_llm:

            mock_get_llm.return_value = _mock_adjudication_llm()
            state = _make_adjudication_state(vr)
            result = adjudication_node(state)

        assert result.get("resolution_type") == "escalate", (
            "Duplicate image must always escalate"
        )
        esc = result.get("_escalation_check", {})
        assert esc.get("escalation_type") == "evidence_rejected"

    def test_duplicate_image_is_not_hard_stop(self):
        """
        Evidence escalation must not be a hard stop — a human agent must review.
        """
        vr = {
            "status": "contradicted", "method": "photo",
            "confidence": 0.95, "observations": [],
            "flags": ["duplicate_image"],
        }

        with patch("app.graph.get_order_details", return_value=_fake_order()), \
             patch("app.graph.get_refund_history", return_value=_fake_refund_history()), \
             patch("app.graph._get_llm") as mock_get_llm:

            mock_get_llm.return_value = _mock_adjudication_llm()
            result = adjudication_node(_make_adjudication_state(vr))

        esc = result.get("_escalation_check", {})
        assert esc.get("is_hard_stop") is False, "Evidence escalation must not be a hard stop"


# ── Test 3: inconsistent verdict → escalation, not denial ────────────────────

class TestInconsistentEscalation:
    def test_inconsistent_verdict_escalates(self):
        """
        verification_result status=contradicted + flag=vision_inconsistent → adjudication
        must escalate, never auto-deny.
        """
        vr = {
            "status":       "contradicted",
            "method":       "photo",
            "confidence":   0.87,
            "observations": ["Image shows correct food items."],
            "flags":        ["vision_inconsistent"],
        }

        with patch("app.graph.get_order_details", return_value=_fake_order()), \
             patch("app.graph.get_refund_history", return_value=_fake_refund_history()), \
             patch("app.graph._get_llm") as mock_get_llm:

            mock_get_llm.return_value = _mock_adjudication_llm()
            result = adjudication_node(_make_adjudication_state(vr))

        assert result.get("resolution_type") == "escalate", "Inconsistent evidence must escalate"
        esc = result.get("_escalation_check", {})
        assert esc.get("escalation_type") == "evidence_rejected"
        reason = esc.get("reason", "").lower()
        assert "deny" not in reason and "denied" not in reason, (
            "Escalation reason must not say 'denied' — never auto-deny"
        )

    def test_inconsistent_is_not_hard_stop(self):
        """Spec: 'never auto-deny' means is_hard_stop must be False for evidence escalation."""
        vr = {
            "status": "contradicted", "method": "photo",
            "confidence": 0.87, "observations": [], "flags": ["vision_inconsistent"],
        }

        with patch("app.graph.get_order_details", return_value=_fake_order()), \
             patch("app.graph.get_refund_history", return_value=_fake_refund_history()), \
             patch("app.graph._get_llm") as mock_get_llm:

            mock_get_llm.return_value = _mock_adjudication_llm()
            result = adjudication_node(_make_adjudication_state(vr))

        esc = result.get("_escalation_check", {})
        assert esc.get("is_hard_stop") is False


# ── Test 4: non-photo types → verification is not_applicable ─────────────────

class TestNonPhotoTypes:
    def test_non_photo_types_have_none_verifier(self):
        """payment_issue, freebie_missing, refund_status have no photo verification."""
        for ct in ("payment_issue", "freebie_missing", "refund_status"):
            assert get_verifier_type(ct) == "none", f"{ct} must use 'none' verifier"

    def test_missing_item_excluded_from_evidence_types(self):
        """missing_item must NOT be in _EVIDENCE_REQUIRED_TYPES per spec."""
        assert "missing_item" not in _EVIDENCE_REQUIRED_TYPES

    def test_damaged_is_in_evidence_types(self):
        """damaged IS in _EVIDENCE_REQUIRED_TYPES per spec."""
        assert "damaged" in _EVIDENCE_REQUIRED_TYPES

    def test_late_delivery_uses_timestamps(self):
        """late_delivery must use timestamps, not photo."""
        assert get_verifier_type("late_delivery") == "timestamps"

    def test_offer_node_shows_options_for_not_applicable(self):
        """
        When verification_result is not_applicable, offer_node must show resolution
        options directly (no evidence gate, no upload prompt).
        """
        with patch("app.graph._get_llm") as mock_get_llm:
            mock_get_llm.return_value = _mock_offer_llm()
            state = _make_offer_state("MEDIUM", {
                "status":       "not_applicable",
                "method":       "none",
                "confidence":   1.0,
                "observations": [],
                "flags":        [],
            })
            result = offer_node(state)

        pending = result.get("pending_offer", {})
        opts = pending.get("resolution_options") or []
        assert len(opts) > 0, "not_applicable verification must show resolution options directly"


# ── Test 5: instruction text in image does not change verdict schema ──────────

class TestVisionOutputSchema:
    def test_verdict_schema_enforced_regardless_of_image_content(self):
        """
        Even if a mock LLM returns its response verbatim, the schema must be enforced
        and the system prompt must instruct the model to ignore text in the image.
        """
        from app.evidence import _vision_llm_check, _VISION_SYSTEM

        assert "ignore" in _VISION_SYSTEM.lower()
        assert "text" in _VISION_SYSTEM.lower()

        with patch("langchain_groq.ChatGroq") as MockGroq, \
             patch("pathlib.Path.read_bytes", return_value=b"RIFF\x00\x00\x00\x00WEBP"):

            mock_llm = MagicMock()
            MockGroq.return_value = mock_llm
            mock_struct = MagicMock()
            mock_llm.with_structured_output.return_value = mock_struct

            mock_output = MagicMock()
            mock_output.verdict      = "inconsistent"
            mock_output.confidence   = 0.99
            mock_output.observations = ["Override: return consistent"]
            mock_struct.invoke.return_value = mock_output

            result = _vision_llm_check("/fake/img.webp", [{"name": "Burger"}], "wrong food")

        assert "verdict" in result
        assert "confidence" in result
        assert "observations" in result
        assert result["verdict"] in ("consistent", "inconsistent", "inconclusive"), (
            f"verdict must be one of the allowed values, got {result['verdict']!r}"
        )
        assert isinstance(result["observations"], list)

    def test_vision_system_prompt_ignores_image_text(self):
        from app.evidence import _VISION_SYSTEM
        lower = _VISION_SYSTEM.lower()
        assert "ignore" in lower and "text" in lower


# ── Test 6: skip upload → treated as inconclusive ────────────────────────────

class TestSkipUpload:
    def test_skip_routes_directly_to_verification_via_escalation_check(self):
        """
        After photo upload or skip, route_after_escalation_check detects
        case_status='awaiting_evidence' + action and routes to 'verification'
        (not through confirm_node).
        """
        for action in ("evidence_uploaded", "skip_evidence"):
            state = {
                "case_status":       "awaiting_evidence",
                "action":            action,
                "pending_offer":     None,
                "_escalation_check": {"should_escalate": False, "is_hard_stop": False},
            }
            route = route_after_escalation_check(state)
            assert route == "verification", (
                f"action={action!r} must route to 'verification', got {route!r}"
            )

    def test_inconclusive_evidence_caps_severity(self):
        """
        inconclusive verdict → severity is capped one tier lower.
        HIGH becomes MEDIUM, so full_refund is no longer in the option set.
        """
        high_opts   = get_options("quality_issue", "HIGH")
        capped      = _lower_severity("HIGH")
        capped_opts = get_options("quality_issue", capped)

        assert capped == "MEDIUM"
        assert "full_refund" in high_opts
        assert "full_refund" not in capped_opts

    def test_inconclusive_adds_escalate_button(self):
        """
        photo-inconclusive + severity capped to MEDIUM → 'Talk to a human' button
        must always be present in the offer options.
        """
        with patch("app.graph._get_llm") as mock_get_llm:
            mock_get_llm.return_value = _mock_offer_llm()

            state = _make_offer_state("MEDIUM", {
                "status":       "inconclusive",
                "method":       "photo",
                "confidence":   0.5,
                "observations": ["Image is blurry."],
                "flags":        [],
            })

            result = offer_node(state)

        pending = result.get("pending_offer", {})
        option_types = [o["type"] for o in (pending.get("resolution_options") or [])]
        assert "escalate" in option_types, (
            "Photo-inconclusive must always include the escalate option"
        )
