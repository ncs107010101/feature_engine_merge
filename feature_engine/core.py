"""
feature_engine.core
================================
Base classes and registry mechanism for the feature calculation module.
"""

from abc import ABC, abstractmethod
from typing import Any, Union, Dict, Type, List
import pandas as pd


# ---------------------------------------------------------------------------
# Feature Registry (Singleton)
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """
    A global registry that maps feature names to their implementing classes.
    """

    _features: Dict[str, Type["BaseFeature"]] = {}

    @classmethod
    def register(cls, feature_cls: Type["BaseFeature"]) -> Type["BaseFeature"]:
        name = feature_cls.name
        if name in cls._features:
            raise ValueError(
                f"Feature '{name}' is already registered by "
                f"{cls._features[name].__name__}. "
                f"Cannot register {feature_cls.__name__} with the same name."
            )
        cls._features[name] = feature_cls
        return feature_cls

    @classmethod
    def get(cls, name: str) -> Type["BaseFeature"]:
        if name not in cls._features:
            raise KeyError(f"Feature '{name}' is not registered.")
        return cls._features[name]

    @classmethod
    def get_all(cls) -> Dict[str, Type["BaseFeature"]]:
        return dict(cls._features)
        
    @classmethod
    def get_by_combination(cls, combination: str) -> Dict[str, Type["BaseFeature"]]:
        return {k: v for k, v in cls._features.items() if v.data_combination == combination}

    @classmethod
    def list_names(cls) -> List[str]:
        return sorted(cls._features.keys())

    @classmethod
    def clear(cls) -> None:
        cls._features.clear()


# ---------------------------------------------------------------------------
# Decorator Shorthand
# ---------------------------------------------------------------------------

def register_feature(cls: Type["BaseFeature"]) -> Type["BaseFeature"]:
    return FeatureRegistry.register(cls)


# ---------------------------------------------------------------------------
# Base Feature
# ---------------------------------------------------------------------------

class BaseFeature(ABC):
    """
    Abstract base class that all features must inherit from.
    """

    # --- Class-level attributes (to be overridden) ---
    name: str = ""
    description: str = ""
    required_columns: List[str] = []
    data_combination: str = "" # e.g. "single_stock_daily", "single_stock_tick", "single_stock_broker", "cross_broker_tick"

    # --- Public API ---

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs: Any) -> Union[pd.Series, pd.DataFrame]:
        ...

    def validate(self, data: pd.DataFrame) -> bool:
        missing = [c for c in self.required_columns if c not in data.columns]
        if missing:
            raise ValueError(
                f"Feature '{self.name}' requires columns {self.required_columns}, "
                f"but the following are missing: {missing}. "
                f"Available columns: {list(data.columns)}"
            )
        return True

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(name='{self.name}', "
            f"data_combination='{self.data_combination}')>"
        )
