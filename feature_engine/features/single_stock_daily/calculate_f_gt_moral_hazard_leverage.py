import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

EPS = 1e-8

@register_feature
class FeatureGtMoralHazardLeverage(BaseFeature):
    name = "f_gt_moral_hazard_leverage"
    description = "Game Theory Module 5: Moral hazard leverage. Retail leveraged up while large holders exit."
    required_columns = ["StockId", "Date", "_dt_pct", "_margin_usage", "_large_holder_pct"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # raw = (_dt_pct + _margin_usage) / (_large_holder_pct + eps)
        raw = (df["_dt_pct"] + df["_margin_usage"]) / (df["_large_holder_pct"] + EPS)
        
        # zscore_rolling(60)
        out = zscore_rolling(raw, 60)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
