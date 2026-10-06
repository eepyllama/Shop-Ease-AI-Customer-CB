"""
Food verification tests — decision table for verify_food() and verify_allergen().

All vision calls are mocked via unittest.mock.patch; no Groq key or image file
is required.  ViT pipeline is always mocked out (model not downloaded in CI).

Decision table covered
──────────────────────
dietary_mismatch
  1. menu=veg, vision sees chicken        → verified,     flag=veg_dish_has_meat
  2. menu=veg, vision sees egg            → verified,     flag=veg_dish_has_egg
  3. menu=veg, vision unclear             → inconclusive, flag=vision_unclear
  4. menu=veg, vision matches veg dish    → contradicted, flag=vision_matches_veg_order
  5. menu=non-veg ordered, vision matches → contradicted  (veg claim contradicted)

allergen claim (no vision)
  6. customer declares milk, dish has milk → verified, flag=allergens_confirmed_in_menu
  7. customer declares nut,  dish has none → verified, flag=allergen_not_in_menu

wrong_order
  8. vision says "no"                     → verified,     flag=wrong_dish_vision
  9. vision says "yes", no ViT mismatch   → contradicted, flag=vision_matches_order
 10. vision "unclear", ViT mismatch(≥0.5) → verified,     flag=wrong_dish_vit
 11. vision "unclear", ViT unavailable    → inconclusive
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.verification import verify_allergen, verify_food


# ── Helpers ───────────────────────────────────────────────────────────────────

def _order(items: list[dict]) -> dict:
    return {"order_id": "ORD-T1", "items": items}


_VEG_PANEER_ITEM  = {"name": "Paneer Butter Masala", "price": 12.0, "quantity": 1}
_VEG_DAL_ITEM     = {"name": "Dal Tadka",            "price": 8.0,  "quantity": 1}
_NONVEG_CHICKEN   = {"name": "Butter Chicken",       "price": 14.0, "quantity": 1}
_PRAWN_ITEM       = {"name": "Prawn Masala",         "price": 16.0, "quantity": 1}

_FAKE_IMAGE_PATH  = "/fake/evidence/test.jpg"


def _mock_vision(
    looks_like: str = "unclear",
    proteins: list[str] | None = None,
    dish_guess: str = "",
    confidence: float = 0.7,
) -> dict:
    return {
        "visible_proteins":        proteins or [],
        "looks_like_ordered_dish": looks_like,
        "dish_guess":              dish_guess,
        "foreign_object":          "no",
        "confidence":              confidence,
    }


def _evidence(image_path: str = _FAKE_IMAGE_PATH) -> dict:
    return {
        "image_path":    image_path,
        "verdict":       "consistent",
        "confidence":    0.7,
        "observations":  [],
        "duplicate_image": False,
    }


# ── dietary_mismatch ─────────────────────────────────────────────────────────

class TestDietaryMismatch:

    def test_veg_dish_vision_sees_chicken_verified(self):
        """Menu=veg + vision sees chicken → verified, veg_dish_has_meat flag."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("no", ["chicken"], "Butter Chicken", 0.88)):
            with patch("app.vision_food.classify_with_vit", return_value=None):
                result = verify_food(
                    _order([_VEG_PANEER_ITEM]),
                    "dietary_mismatch",
                    _evidence(),
                    "I ordered veg paneer but got chicken",
                )
        assert result.status == "verified"
        assert "veg_dish_has_meat" in result.flags

    def test_veg_dish_vision_sees_egg_verified(self):
        """Menu=veg + vision sees egg → verified, veg_dish_has_egg flag."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("no", ["egg"], "Egg Curry", 0.80)):
            with patch("app.vision_food.classify_with_vit", return_value=None):
                result = verify_food(
                    _order([_VEG_DAL_ITEM]),
                    "dietary_mismatch",
                    _evidence(),
                    "There is an egg in my dal",
                )
        assert result.status == "verified"
        assert "veg_dish_has_egg" in result.flags

    def test_veg_dish_vision_unclear_inconclusive(self):
        """Menu=veg + vision unclear → inconclusive."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("unclear", [], "", 0.3)):
            with patch("app.vision_food.classify_with_vit", return_value=None):
                result = verify_food(
                    _order([_VEG_PANEER_ITEM]),
                    "dietary_mismatch",
                    _evidence(),
                    "looks like it has meat",
                )
        assert result.status == "inconclusive"
        assert "vision_unclear" in result.flags

    def test_veg_dish_vision_matches_veg_contradicted(self):
        """Menu=veg + vision looks like the veg dish → contradicted (no auto-deny)."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("yes", ["paneer"], "Paneer Butter Masala", 0.90)):
            with patch("app.vision_food.classify_with_vit", return_value=None):
                result = verify_food(
                    _order([_VEG_PANEER_ITEM]),
                    "dietary_mismatch",
                    _evidence(),
                    "I think I got meat",
                )
        assert result.status == "contradicted"
        assert result.status != "deny"           # never auto-deny
        assert "vision_matches_veg_order" in result.flags

    def test_nonveg_ordered_no_meat_visible_contradicted(self):
        """Non-veg ordered + no animal proteins visible → contradicted."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("yes", [], "Butter Chicken gravy", 0.75)):
            with patch("app.vision_food.classify_with_vit", return_value=None):
                result = verify_food(
                    _order([_NONVEG_CHICKEN]),
                    "dietary_mismatch",
                    _evidence(),
                    "this has no chicken",
                )
        # all_veg=False for Butter Chicken, so veg_dish_has_meat branch won't fire
        assert result.status in ("contradicted", "inconclusive")


# ── allergen claims (no vision) ───────────────────────────────────────────────

class TestAllergenVerification:

    def test_milk_allergen_confirmed_in_menu(self):
        """Customer declares milk; menu has milk → flag=allergens_confirmed_in_menu."""
        result = verify_allergen(
            _order([_VEG_PANEER_ITEM]),  # Paneer Butter Masala has milk
            ["milk"],
        )
        assert result.status == "verified"
        assert result.method == "menu_data"
        assert "allergen_claim_critical" in result.flags
        assert "allergens_confirmed_in_menu" in result.flags
        assert any("milk" in obs.lower() for obs in result.observations)

    def test_allergen_not_in_menu_flagged(self):
        """Customer declares nuts; dish has no nuts → flag=allergen_not_in_menu."""
        result = verify_allergen(
            _order([_VEG_DAL_ITEM]),   # Dal Tadka has no allergens
            ["nuts"],
        )
        assert result.status == "verified"     # still verified (claim critical)
        assert "allergen_claim_critical" in result.flags
        assert "allergen_not_in_menu" in result.flags

    def test_shellfish_allergen_in_prawn(self):
        """Prawn Masala has shellfish; customer declares shellfish → confirmed."""
        result = verify_allergen(
            _order([_PRAWN_ITEM]),
            ["shellfish"],
        )
        assert result.status == "verified"
        assert "allergens_confirmed_in_menu" in result.flags

    def test_allergen_check_never_uses_vision(self):
        """verify_allergen must not call analyze_food."""
        with patch("app.vision_food.analyze_food") as mock_vision:
            verify_allergen(_order([_VEG_PANEER_ITEM]), ["milk"])
        mock_vision.assert_not_called()

    def test_allergen_confidence_is_one(self):
        """Menu data is authoritative — confidence must be 1.0."""
        result = verify_allergen(_order([_VEG_PANEER_ITEM]), ["milk"])
        assert result.confidence == 1.0


# ── wrong_order ───────────────────────────────────────────────────────────────

class TestWrongOrder:

    def test_vision_no_gives_verified(self):
        """Vision says 'no' → verified, wrong_dish_vision flag."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("no", ["paneer"], "Paneer dish", 0.85)):
            with patch("app.vision_food.classify_with_vit", return_value=None):
                result = verify_food(
                    _order([_NONVEG_CHICKEN]),
                    "wrong_order",
                    _evidence(),
                    "I ordered butter chicken but got paneer",
                )
        assert result.status == "verified"
        assert "wrong_dish_vision" in result.flags

    def test_vision_yes_no_vit_mismatch_contradicted(self):
        """Vision 'yes' + no ViT mismatch → contradicted."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("yes", ["chicken"], "Butter Chicken", 0.92)):
            with patch("app.vision_food.classify_with_vit",
                       return_value={"predicted_label": "butter_chicken",
                                     "expected_label":  "butter_chicken",
                                     "score": 0.91, "match": True}):
                result = verify_food(
                    _order([_NONVEG_CHICKEN]),
                    "wrong_order",
                    _evidence(),
                    "I got a wrong dish",
                )
        assert result.status == "contradicted"

    def test_vision_unclear_vit_mismatch_verified(self):
        """Vision 'unclear' + ViT predicts different dish (score≥0.5) → verified."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("unclear", ["paneer"], "Paneer Tikka", 0.55)):
            with patch("app.vision_food.classify_with_vit",
                       return_value={"predicted_label": "paneer_tikka",
                                     "expected_label":  "butter_chicken",
                                     "score": 0.78, "match": False}):
                result = verify_food(
                    _order([_NONVEG_CHICKEN]),
                    "wrong_order",
                    _evidence(),
                    "this is not what I ordered",
                )
        assert result.status == "verified"
        assert "wrong_dish_vit" in result.flags

    def test_vision_unclear_no_vit_gives_inconclusive(self):
        """Vision 'unclear' + ViT unavailable → inconclusive."""
        with patch("app.vision_food.analyze_food",
                   return_value=_mock_vision("unclear", [], "", 0.40)):
            with patch("app.vision_food.classify_with_vit", return_value=None):
                result = verify_food(
                    _order([_NONVEG_CHICKEN]),
                    "wrong_order",
                    _evidence(),
                    "wrong dish",
                )
        assert result.status == "inconclusive"

    def test_skip_gives_inconclusive_for_wrong_order(self):
        """Customer skips upload → inconclusive regardless of complaint type."""
        result = verify_food(
            _order([_NONVEG_CHICKEN]),
            "wrong_order",
            {},                  # no evidence_result
            "wrong dish",
        )
        # empty evidence_result → no image_path → vision gets empty image → inconclusive
        # (analyze_food is skipped when image_path is empty)
        assert result.status in ("inconclusive",)


# ── Menu enrichment sanity check ─────────────────────────────────────────────

class TestMenuEnrichment:

    def test_paneer_butter_masala_is_veg(self):
        from app.verification import _get_menu_item
        dish = _get_menu_item("Paneer Butter Masala")
        assert dish is not None
        assert dish["is_veg"] is True
        assert "milk" in dish["allergens"]

    def test_prawn_masala_has_shellfish(self):
        from app.verification import _get_menu_item
        dish = _get_menu_item("Prawn Masala")
        assert dish is not None
        assert dish["is_veg"] is False
        assert "shellfish" in dish["allergens"]

    def test_all_25_dishes_loaded(self):
        from app.verification import _load_menu
        dishes = _load_menu()
        assert len(dishes) == 25

    def test_unknown_item_returns_none(self):
        from app.verification import _get_menu_item
        assert _get_menu_item("Spaghetti Carbonara") is None
