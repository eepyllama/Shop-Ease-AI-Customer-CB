"""
Severity classifier — translates (complaint_type, context, item_attributes) → severity level.

Rules:
  1. allergen_safety path: customer declared allergen AND order contains it → CRITICAL,
     allergen_safety=True.  This path is SEPARATE from dietary_mismatch and must NOT
     route through it.
  2. dietary_mismatch path: customer stated veg/halal/etc restriction AND a delivered
     item violates it → HIGH.
  3. quality_issue with food-safety keywords → HIGH.
  4. Otherwise: classify by delay band or claim value against LOW/MEDIUM/HIGH thresholds.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Allergen and dietary keyword sets ────────────────────────────────────────

_ALLERGEN_KEYWORDS = {
    "peanut", "peanuts", "nut", "nuts", "tree nut", "tree nuts",
    "dairy", "milk", "lactose", "cheese",
    "gluten", "wheat", "flour",
    "soy", "soya",
    "shellfish", "shrimp", "prawn", "crab", "lobster",
    "fish", "salmon", "tuna",
    "egg", "eggs",
    "sesame", "mustard", "sulphite", "sulphites",
}

_DIETARY_RESTRICTION_KEYWORDS = {
    "vegetarian", "veg", "vegan",
    "halal", "kosher", "jain",
    "no meat", "no beef", "no pork", "no chicken", "no fish",
    "plant-based", "plant based",
}

_FOOD_SAFETY_KEYWORDS = {
    "inedible", "foreign object", "hair", "glass", "plastic",
    "mold", "mould", "rotten", "spoiled", "expired",
    "food poisoning", "sick", "allergic reaction", "vomiting",
}

# Delay bands → severity (minutes over promised time)
_DELAY_BANDS = [
    (15, "LOW"),
    (30, "MEDIUM"),
    (90, "HIGH"),
]


def _normalize(text: str) -> str:
    return text.lower()


def _declared_allergens(text: str) -> list[str]:
    """Extract allergen types explicitly declared in a customer message."""
    norm = _normalize(text)
    found = []
    for kw in _ALLERGEN_KEYWORDS:
        # Require the word to appear in a sensitivity/allergy context
        pattern = r"\b(allerg|intoleran|sensitiv|cannot eat|can.t eat|avoid)\b.{0,40}\b" + re.escape(kw) + r"\b"
        if re.search(pattern, norm):
            found.append(kw)
        # Also catch: "I'm allergic to X" / "X allergy"
        pattern2 = r"\b" + re.escape(kw) + r"\b.{0,20}\b(allerg|intoleran|sensitiv)\b"
        if re.search(pattern2, norm):
            found.append(kw)
    return list(set(found))


def _declared_dietary_restriction(text: str) -> Optional[str]:
    """Return the dietary restriction mentioned in the message, or None."""
    norm = _normalize(text)
    for kw in _DIETARY_RESTRICTION_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", norm):
            return kw
    return None


def _item_contains_allergen(item: dict, allergen: str) -> bool:
    item_allergens = [a.lower() for a in item.get("allergens", [])]
    return allergen.lower() in item_allergens


def _item_violates_diet(item: dict, restriction: str) -> bool:
    norm = restriction.lower()
    if norm in ("vegetarian", "veg", "vegan", "jain"):
        return not item.get("veg", True)
    if norm in ("halal",):
        # Conservative: any non-veg item is suspect without halal certification
        return not item.get("veg", True)
    if norm in ("kosher",):
        return not item.get("veg", True)
    if norm in ("no pork",):
        name = item.get("name", "").lower()
        return "pork" in name or "bacon" in name or "ham" in name
    return False


def classify_severity(
    complaint_type: str,
    claim_description: str,
    items: list[dict],
    dietary_restriction: Optional[str],
    allergens_declared: list[str],
    delay_minutes: int = 0,
) -> tuple[str, bool]:
    """
    Returns (severity, allergen_safety).

    allergen_safety=True means an allergen declared by the customer is present in
    the delivered items.  This flag bypasses all cost guardrails.
    The allergen path does NOT trigger dietary_mismatch logic — they are separate.
    """

    # ── 1. Allergen safety (CRITICAL, separate path) ─────────────────────────
    if allergens_declared:
        for allergen in allergens_declared:
            for item in items:
                if _item_contains_allergen(item, allergen):
                    return "CRITICAL", True

    # ── 2. Dietary mismatch (HIGH, separate from allergen) ───────────────────
    # When complaint_type is dietary_mismatch and the order contains a non-veg
    # item, that's HIGH regardless of whether the customer stated a restriction.
    if complaint_type == "dietary_mismatch":
        if any(item.get("veg") is False for item in items):
            return "HIGH", False

    if dietary_restriction:
        for item in items:
            if _item_violates_diet(item, dietary_restriction):
                return "HIGH", False

    # ── 3. Quality issue with food-safety keywords → HIGH ────────────────────
    if complaint_type == "quality_issue":
        desc_norm = _normalize(claim_description)
        for kw in _FOOD_SAFETY_KEYWORDS:
            if kw in desc_norm:
                return "HIGH", False
        # Minor quality issue without safety words
        return "LOW", False

    # ── 4. Late delivery — classify by delay band ────────────────────────────
    if complaint_type == "late_delivery":
        severity = "LOW"
        for threshold, level in _DELAY_BANDS:
            if delay_minutes >= threshold:
                severity = level
        if delay_minutes >= 90:
            severity = "HIGH"
        return severity, False

    # ── 5. Wrong order ────────────────────────────────────────────────────────
    if complaint_type == "wrong_order":
        return "MEDIUM", False

    # ── 6. Missing item ───────────────────────────────────────────────────────
    if complaint_type == "missing_item":
        return "MEDIUM", False

    # ── 7. Default ────────────────────────────────────────────────────────────
    return "LOW", False


def extract_intake_context(message: str) -> dict:
    """
    Extract dietary_restriction and allergens_declared from the raw customer message.
    Called by intake_node before the LLM call so the LLM prompt has context.
    """
    return {
        "dietary_restriction": _declared_dietary_restriction(message),
        "allergens_declared": _declared_allergens(message),
    }
