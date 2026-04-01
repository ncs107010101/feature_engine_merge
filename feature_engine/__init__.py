"""
feature_engine
==============
A modular feature calculation framework.
"""

from .api import compute_features, list_features, describe_features
from .core import register_feature, BaseFeature

# Trigger auto-discovery
from . import features
