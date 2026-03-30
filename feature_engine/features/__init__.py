"""
feature_engine.features
=====================================
Auto-discovery of feature modules across data combinations.
"""

import importlib
import pkgutil
from pathlib import Path


def _auto_import_features() -> None:
    """Import all sibling modules so that @register_feature decorators fire."""
    package_dir = Path(__file__).resolve().parent

    # Traverse subdirectories for combination groups
    for item in package_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            comb_name = item.name
            for module_info in pkgutil.iter_modules([str(item)]):
                if module_info.name.startswith("_"):
                    continue
                # Import: feature_engine.features.single_stock_daily.calculate_f_sma_20
                importlib.import_module(f".{comb_name}.{module_info.name}", package=__name__)


_auto_import_features()
