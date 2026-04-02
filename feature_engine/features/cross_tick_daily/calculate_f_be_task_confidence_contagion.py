import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureBeTaskConfidenceContagion(BaseFeature):
    name = "f_be_task_confidence_contagion"
    description = "yesterday_success × gap_down × am_retail_buy, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_first500_small_buy", "raw_ret1_shift1", "raw_gap_down"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        df["raw_first500_small_buy"] = df["raw_first500_small_buy"].fillna(0)
        df["raw_ret1_shift1"] = df["raw_ret1_shift1"].fillna(0)
        df["raw_gap_down"] = df["raw_gap_down"].fillna(0)
        
        yesterday_success = (df["raw_ret1_shift1"] > 0.05).astype(int)
        gap_down = df["raw_gap_down"]
        
        raw = yesterday_success * gap_down * df["raw_first500_small_buy"]
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
