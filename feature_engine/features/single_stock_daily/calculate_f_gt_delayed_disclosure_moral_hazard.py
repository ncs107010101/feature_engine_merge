import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

EPS = 1e-8

@register_feature
class FeatureGtDelayedDisclosureMoralHazard(BaseFeature):
    name = "f_gt_delayed_disclosure_moral_hazard"
    description = "Game Theory Module 3: Delayed disclosure moral hazard. Low volatility facade while short interest surges."
    required_columns = ["StockId", "Date", "_vol_20d", "_short_change_5d"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # inv_vol = 1 / (vol_20d + eps)
        inv_vol = 1.0 / (df["_vol_20d"] + EPS)
        
        # raw = inv_vol * short_change_5d
        raw = inv_vol * df["_short_change_5d"]
        
        # zscore_rolling(60)
        out = zscore_rolling(raw, 60)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
