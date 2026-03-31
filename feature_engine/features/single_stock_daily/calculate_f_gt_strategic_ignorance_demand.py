import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureGtStrategicIgnoranceDemand(BaseFeature):
    name = "f_gt_strategic_ignorance_demand"
    description = "Game Theory Module 5: Strategic ignorance demand. Retail increases margin despite falling below SMA20."
    required_columns = ["StockId", "Date", "收盤價", "_sma_20", "_ret_5d", "_margin_change_3"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # is_below_ma = (收盤價 < _sma_20).astype(int)
        is_below_ma = (df["收盤價"] < df["_sma_20"]).astype(int)
        
        # is_falling = (_ret5d < -0.05).astype(int)
        is_falling = (df["_ret_5d"] < -0.05).astype(int)
        
        # is_margin_up = (_margin_change_3 > 0).astype(int)
        is_margin_up = (df["_margin_change_3"] > 0).astype(int)
        
        # raw = is_below_ma * is_falling * is_margin_up * _margin_change_3
        raw = is_below_ma * is_falling * is_margin_up * df["_margin_change_3"]
        
        # zscore_rolling(60)
        out = zscore_rolling(raw, 60)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
