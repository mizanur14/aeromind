"""Leakage-safe feature engineering for C-MAPSS RUL prediction."""

from aeromind.features.pipeline import FeaturePipeline
from aeromind.features.rolling import trailing_features
from aeromind.features.targets import cap_rul, make_target

__all__ = ["FeaturePipeline", "cap_rul", "make_target", "trailing_features"]
