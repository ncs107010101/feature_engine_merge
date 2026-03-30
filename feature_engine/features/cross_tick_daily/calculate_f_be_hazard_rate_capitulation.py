import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureBeHazardRateCapitulation(BaseFeature):
    name = "f_be_hazard_rate_capitulation"
    description = "開盤跳空低開(開盤<前日收盤*0.98) × 前5個Bin的小單賣出總額, ewm_then_zscore(5, 20)"
    required_columns = ["StockId", "Date", "raw_gap_down", "raw_instant_exit"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_gap_down"] * df["raw_instant_exit"]
        smoothed = raw.groupby(df["StockId"]).transform(lambda x: x.ewm(span=5, min_periods=5).mean())
        out = smoothed.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })