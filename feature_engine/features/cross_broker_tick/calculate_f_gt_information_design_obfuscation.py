"""
calculate_f_gt_information_design_obfuscation.py
Group 7: Game Theory - Information Design Obfuscation

Feature: f_gt_information_design_obfuscation
Theory: Creating noise (PrFlag flips, size variance) to mask top5 net selling
Direction: Positive → Noise masking distribution → Extreme LOW return
Data: cross_broker_tick (trade_level1_data + broker_data)
"""

import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import EPS

WINDOW = 20
WIN_EWM = 10


def zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / (rolling_std + EPS)."""
    rm = series.rolling(window, min_periods=max(1, window // 2)).mean()
    rs = series.rolling(window, min_periods=max(1, window // 2)).std()
    return (series - rm) / (rs + EPS)


@register_feature
class FGtInformationDesignObfuscation(BaseFeature):
    name = "f_gt_information_design_obfuscation"
    description = "Information Design Obfuscation: Creates noise (PrFlag flips, size variance) to mask top5 net selling. Direction: Positive → Noise masking distribution → Extreme LOW return."
    required_columns = [
        "StockId", "Date", "raw_obfuscation_index", "Top5_NetSell"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        raw = all_features["raw_obfuscation_index"] * all_features["Top5_NetSell"]
        raw_ewm = raw.ewm(span=WIN_EWM, min_periods=1).mean()
        out = zscore_rolling(raw_ewm, WINDOW)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
