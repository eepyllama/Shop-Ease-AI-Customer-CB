"""
Cross-check verification layer — runs between intake and adjudication.

Provides:
  - VerificationResult dataclass (verdict + metadata)
  - A complaint_type → verifier_type registry
  - Timestamp verifier for late_delivery
  - Photo verifiers delegate to app/evidence.py (phash, EXIF, vision LLM)
  - Food-specific verifiers:
      verify_allergen()  — menu-data only, no vision; always surfaces CRITICAL data
      verify_food()      — dietary_mismatch + wrong_order via vision_food.py

Status values:
  verified        — claim is supported by objective data / evidence
  contradicted    — data or evidence directly contradicts the claim
  inconclusive    — could not confirm or deny (blurry image, missing data, skipped)
  not_applicable  — complaint type has no automatic verification method
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Photo-eligible complaint types ────────────────────────────────────────────
_PHOTO_TYPES: frozenset[str] = frozenset({
    "wrong_order",
    "dietary_mismatch",
    "foreign_object",
    "damaged",
    "quality_issue",
})

# ── Timestamp-eligible complaint types ────────────────────────────────────────
_TIMESTAMP_TYPES: frozenset[str] = frozenset({"late_delivery"})

# ── None / data-lookup only ────────────────────────────────────────────────────
_NONE_TYPES: frozenset[str] = frozenset({"freebie_missing", "payment_issue", "refund_status"})


def get_verifier_type(complaint_type: str) -> str:
    """Return 'photo', 'timestamps', or 'none'."""
    if complaint_type in _PHOTO_TYPES:
        return "photo"
    if complaint_type in _TIMESTAMP_TYPES:
        return "timestamps"
    return "none"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    status: str                          # verified | contradicted | inconclusive | not_applicable
    method: str                          # photo | timestamps | none
    confidence: float = 0.0
    observations: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status":       self.status,
            "method":       self.method,
            "confidence":   self.confidence,
            "observations": list(self.observations),
            "flags":        list(self.flags),
        }

    @classmethod
    def not_applicable(cls) -> "VerificationResult":
        return cls(status="not_applicable", method="none", confidence=1.0)

    @classmethod
    def from_evidence(cls, evidence_result: dict) -> "VerificationResult":
        """Convert an EvidenceResult dict (from app/evidence.py) to a VerificationResult."""
        verdict   = evidence_result.get("verdict", "inconclusive")
        duplicate = evidence_result.get("duplicate_image", False)
        conf      = float(evidence_result.get("confidence", 0.0))
        obs       = list(evidence_result.get("observations", []))

        if duplicate:
            return cls(
                status="contradicted", method="photo", confidence=conf,
                observations=obs, flags=["duplicate_image"],
            )
        if verdict == "inconsistent":
            return cls(
                status="contradicted", method="photo", confidence=conf,
                observations=obs, flags=["vision_inconsistent"],
            )
        if verdict == "consistent":
            return cls(status="verified", method="photo", confidence=conf, observations=obs)
        # inconclusive / unknown
        return cls(status="inconclusive", method="photo", confidence=conf, observations=obs)


# ── Timestamp verifier ─────────────────────────────────────────────────────────

def verify_timestamps(
    order_details: dict,
    claim_description: str,
    *,
    none_min: int = 10,
    low_min: int = 30,
    medium_min: int = 60,
) -> VerificationResult:
    """
    Verify a late_delivery claim against order timestamp data.

    Delay is computed from actual_delivered_at − promised_eta_at_order_time
    (ISO strings) when present; falls back to actual_delivery_min − estimated_delivery_min.

    Returns:
      not_applicable — order not yet delivered, or no timing data
      verified       — data confirms a delay, claim is plausible
      contradicted   — data shows no significant delay despite the claim
      inconclusive   — partial data, can't confirm or deny
    """
    if not order_details:
        return VerificationResult(
            status="not_applicable", method="timestamps",
            observations=["No order data available"],
        )

    # Not-yet-delivered check
    status = order_details.get("status", "")
    actual_delivered_at = order_details.get("actual_delivered_at")
    actual_minutes = order_details.get("actual_delivery_min")

    if actual_delivered_at is None and actual_minutes is None:
        return VerificationResult(
            status="not_applicable", method="timestamps", confidence=1.0,
            observations=["Order not yet delivered."],
            flags=["order_in_progress"],
        )
    if status in ("preparing", "out_for_delivery"):
        return VerificationResult(
            status="not_applicable", method="timestamps", confidence=1.0,
            observations=["Order is still in progress."],
            flags=["order_in_progress"],
        )

    # Compute delay
    promised_eta = order_details.get("promised_eta_at_order_time")
    delay_minutes: int = 0

    if promised_eta and actual_delivered_at:
        try:
            p = datetime.fromisoformat(promised_eta.replace("Z", "+00:00"))
            a = datetime.fromisoformat(actual_delivered_at.replace("Z", "+00:00"))
            delay_minutes = max(0, int((a - p).total_seconds() / 60))
        except (ValueError, AttributeError):
            delay_minutes = 0

    if delay_minutes == 0:
        # Fallback to minute fields
        est = int(order_details.get("estimated_delivery_min") or 0)
        act = int(actual_minutes or 0)
        delay_minutes = max(0, act - est)

    # Map delay to severity tier
    if delay_minutes < none_min:
        data_severity = "none"
    elif delay_minutes < low_min:
        data_severity = "LOW"
    elif delay_minutes < medium_min:
        data_severity = "MEDIUM"
    else:
        data_severity = "HIGH"

    t = claim_description.lower()
    claim_mentions_delay = any(
        w in t for w in ("late", "delay", "slow", "long", "hour", "minutes", "waited")
    )

    # Format timestamps for observations
    promised_str = promised_eta or f"est. {order_details.get('estimated_delivery_min', '?')} min"
    actual_str   = actual_delivered_at or f"actual {actual_minutes} min"
    obs_timing   = f"Promised ETA: {promised_str} · Delivered: {actual_str} · Delay: {delay_minutes} min."

    if data_severity == "none" and claim_mentions_delay:
        return VerificationResult(
            status="contradicted", method="timestamps", confidence=0.90,
            observations=[
                obs_timing,
                "Recorded data does not support a significant delivery delay.",
            ],
            flags=["claim_contradicts_data"],
        )

    if claim_mentions_delay and data_severity != "none":
        return VerificationResult(
            status="verified", method="timestamps", confidence=0.95,
            observations=[obs_timing, f"Delay of {delay_minutes} min confirmed by order records."],
            flags=[f"delay_minutes:{delay_minutes}", f"data_severity:{data_severity}"],
        )

    # No delay-related claim and no significant delay → not applicable
    return VerificationResult(
        status="not_applicable", method="timestamps", confidence=0.5,
        observations=[obs_timing],
    )


# ── Menu helpers ──────────────────────────────────────────────────────────────

_MENU_PATH = Path(__file__).parent.parent / "data" / "menu.json"

# Allergen set validated by menu.json schema
_KNOWN_ALLERGENS: frozenset[str] = frozenset({
    "milk", "nuts", "gluten", "soy", "sesame", "mustard", "shellfish",
})

# Protein tokens that indicate animal-derived content
_MEAT_PROTEINS: frozenset[str] = frozenset({
    "chicken", "mutton", "lamb", "beef", "pork", "fish",
    "prawn", "shrimp", "crab", "lobster", "squid",
})
_EGG_PROTEINS: frozenset[str] = frozenset({"egg", "eggs"})
_ANIMAL_PROTEINS: frozenset[str] = _MEAT_PROTEINS | _EGG_PROTEINS


@lru_cache(maxsize=1)
def _load_menu() -> list[dict]:
    """Load menu.json once and cache."""
    try:
        data = json.loads(_MENU_PATH.read_text(encoding="utf-8"))
        return data.get("dishes", [])
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("verification: cannot load menu.json: %s", exc)
        return []


def _get_menu_item(name: str) -> Optional[dict]:
    """Find a menu dish by name (case-insensitive exact match)."""
    name_lc = name.strip().lower()
    for dish in _load_menu():
        if dish.get("name", "").lower() == name_lc:
            return dish
    return None


def _enrich_ordered_items(order_items: list[dict]) -> list[dict]:
    """
    Match each order item against menu.json and merge in is_veg, allergens,
    typical_visible_proteins.  Items that don't match keep their own fields.
    """
    enriched = []
    for item in order_items:
        dish = _get_menu_item(item.get("name", ""))
        if dish:
            merged = {**item, **dish}   # menu facts override order-level fields
        else:
            merged = dict(item)
        enriched.append(merged)
    return enriched


def _order_allergens(enriched_items: list[dict]) -> list[str]:
    """Return the union of allergens across all enriched order items."""
    seen: set[str] = set()
    for item in enriched_items:
        for a in item.get("allergens", []):
            seen.add(a.lower())
    return sorted(seen)


def _all_veg(enriched_items: list[dict]) -> bool:
    return all(item.get("is_veg", True) for item in enriched_items)


# ── Allergen verification (no vision) ────────────────────────────────────────

def verify_allergen(
    order_details: dict,
    allergens_declared: list[str],
) -> VerificationResult:
    """
    For allergen claims: look up menu data, confirm whether the ordered dish
    actually contains the declared allergens, and surface that data for the
    handoff.  Vision is never consulted.

    Always returns status='verified' with flag='allergen_claim_critical' so
    adjudication escalates immediately.  The menu allergen list is embedded in
    observations for the human-agent handoff.
    """
    items = (order_details or {}).get("items", [])
    enriched = _enrich_ordered_items(items)
    order_allergen_set = set(_order_allergens(enriched))
    declared_set = {a.lower() for a in (allergens_declared or [])}

    confirmed = sorted(order_allergen_set & declared_set)
    present_not_declared = sorted(order_allergen_set - declared_set)
    absent = sorted(declared_set - order_allergen_set)

    obs: list[str] = []
    flags: list[str] = ["allergen_claim_critical"]

    if enriched:
        item_names = ", ".join(i.get("name", "?") for i in enriched)
        obs.append(f"Ordered items: {item_names}.")

    if confirmed:
        obs.append(
            f"Menu data confirms these allergens are present: {', '.join(confirmed)}."
        )
        flags.append("allergens_confirmed_in_menu")
    if present_not_declared:
        obs.append(
            f"Additional allergens in menu data (not declared by customer): "
            f"{', '.join(present_not_declared)}."
        )
    if absent:
        obs.append(
            f"These allergens declared by customer are not listed in menu data: "
            f"{', '.join(absent)}."
        )
        flags.append("allergen_not_in_menu")
    if not order_allergen_set:
        obs.append("No allergens listed in menu data for ordered items.")

    return VerificationResult(
        status="verified",
        method="menu_data",
        confidence=1.0,
        observations=obs,
        flags=flags,
    )


# ── Food vision verification ──────────────────────────────────────────────────

def verify_food(
    order_details: dict,
    complaint_type: str,
    evidence_result: dict,
    claim_description: str = "",
) -> VerificationResult:
    """
    Run food-specific verification for dietary_mismatch and wrong_order.

    Uses app/vision_food.py for the vision model call and, optionally, the ViT
    classifier for wrong_order when the dish is in VIT_LABEL_MAP.

    Decision rules
    ──────────────
    dietary_mismatch
      menu=veg AND vision sees meat/egg → verified  (flag: veg_dish_has_meat)
      vision unclear/inconclusive       → inconclusive
      vision matches veg dish           → contradicted  (escalate, never deny)

    wrong_order
      vision says "no"                  → verified  (flag: wrong_dish_vision)
      ViT predicts a different label    → verified  (flag: wrong_dish_vit)
      vision says "yes" + ViT matches   → contradicted
      vision/ViT unclear                → inconclusive

    Vision may raise severity; allergen/safety claims are handled separately
    by verify_allergen() and are never decided here.
    """
    from app.vision_food import analyze_food, classify_with_vit

    items = (order_details or {}).get("items", [])
    enriched = _enrich_ordered_items(items)
    image_path = evidence_result.get("image_path", "") if evidence_result else ""

    vision: dict = {}
    if image_path:
        vision = analyze_food(image_path, enriched, complaint_type, claim_description)

    looks_like  = vision.get("looks_like_ordered_dish", "unclear")
    proteins    = {p.lower() for p in vision.get("visible_proteins", [])}
    conf        = float(vision.get("confidence", 0.0))
    dish_guess  = vision.get("dish_guess", "")
    obs_base    = f"Vision: looks like ordered dish = '{looks_like}', confidence = {conf:.0%}."
    if dish_guess:
        obs_base += f" Dish guess: '{dish_guess}'."
    if proteins:
        obs_base += f" Visible proteins: {', '.join(sorted(proteins))}."

    if complaint_type == "dietary_mismatch":
        return _decide_dietary_mismatch(enriched, proteins, looks_like, conf, obs_base)

    if complaint_type == "wrong_order":
        ordered_name = enriched[0].get("name", "") if enriched else ""
        vit_result = classify_with_vit(image_path, ordered_name) if image_path else None
        return _decide_wrong_order(looks_like, conf, obs_base, vit_result)

    # Fallback for other photo types — delegate to the generic path
    return VerificationResult.from_evidence(evidence_result or {})


def _decide_dietary_mismatch(
    enriched: list[dict],
    proteins: set[str],
    looks_like: str,
    conf: float,
    obs_base: str,
) -> VerificationResult:
    all_veg = _all_veg(enriched)
    animal_found = bool(proteins & _ANIMAL_PROTEINS)
    meat_found   = bool(proteins & _MEAT_PROTEINS)
    egg_found    = bool(proteins & _EGG_PROTEINS)

    if all_veg and animal_found:
        protein_list = ", ".join(sorted(proteins & _ANIMAL_PROTEINS))
        return VerificationResult(
            status="verified",
            method="photo",
            confidence=conf,
            observations=[
                obs_base,
                f"Menu confirms this is a veg dish. "
                f"Vision detected animal protein(s): {protein_list}.",
            ],
            flags=["veg_dish_has_meat" if meat_found else "veg_dish_has_egg"],
        )

    if looks_like in ("unclear",) or (not animal_found and looks_like == "unclear"):
        return VerificationResult(
            status="inconclusive",
            method="photo",
            confidence=conf,
            observations=[obs_base, "Could not confirm or deny dietary mismatch from photo."],
            flags=["vision_unclear"],
        )

    if looks_like == "no" and not animal_found:
        # Dish doesn't look like what was ordered but no animal protein seen — inconclusive
        return VerificationResult(
            status="inconclusive",
            method="photo",
            confidence=conf,
            observations=[obs_base, "Dish appearance unclear; no animal proteins detected."],
            flags=["vision_unclear"],
        )

    # looks_like == "yes" or "unclear" without animal protein → contradicted
    return VerificationResult(
        status="contradicted",
        method="photo",
        confidence=conf,
        observations=[
            obs_base,
            "Vision does not detect animal proteins; dish appears to match the veg order.",
        ],
        flags=["vision_matches_veg_order"],
    )


def _decide_wrong_order(
    looks_like: str,
    conf: float,
    obs_base: str,
    vit_result: Optional[dict],
) -> VerificationResult:
    flags: list[str] = []
    obs: list[str] = [obs_base]
    vit_mismatch = False

    if vit_result:
        vit_line = (
            f"ViT classifier: predicted '{vit_result['predicted_label']}', "
            f"expected '{vit_result['expected_label']}', "
            f"score {vit_result['score']:.0%}."
        )
        obs.append(vit_line)
        if not vit_result["match"] and vit_result["score"] >= 0.5:
            flags.append("wrong_dish_vit")
            vit_mismatch = True

    if looks_like == "no":
        flags.append("wrong_dish_vision")
        return VerificationResult(
            status="verified", method="photo", confidence=conf,
            observations=obs + ["Vision confirms dish does not match the order."],
            flags=flags,
        )

    if vit_mismatch and looks_like in ("unclear", "no"):
        return VerificationResult(
            status="verified", method="photo", confidence=conf,
            observations=obs + ["ViT classifier confirms a different dish was delivered."],
            flags=flags,
        )

    if looks_like == "yes":
        if not vit_mismatch:
            return VerificationResult(
                status="contradicted", method="photo", confidence=conf,
                observations=obs + ["Vision and classifier both suggest the correct dish was delivered."],
                flags=flags or ["vision_matches_order"],
            )

    # Default: inconclusive
    return VerificationResult(
        status="inconclusive", method="photo", confidence=conf,
        observations=obs + ["Could not conclusively confirm or deny wrong-order claim."],
        flags=flags or ["vision_unclear"],
    )
