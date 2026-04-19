"""Promotion utilities for model release decisions.

@meta
name: promotion
type: module
domain: promotion
responsibility:
  - Provide promotion behavior for `src/promotion/__init__.py`.
inputs: []
outputs: []
tags:
  - promotion
lifecycle:
  status: active
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from config.runtime import PromotionConfig, load_promotion_config

if TYPE_CHECKING:
    from .promote_model import PromotionDecision

__all__ = ["PromotionConfig", "PromotionDecision", "evaluate_promotion", "load_promotion_config"]


def __getattr__(name: str) -> Any:
    if name in {"PromotionDecision", "evaluate_promotion"}:
        from .promote_model import PromotionDecision, evaluate_promotion

        exports = {
            "PromotionDecision": PromotionDecision,
            "evaluate_promotion": evaluate_promotion,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
