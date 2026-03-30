import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FCoreSourceContributionRatio(BaseFeature):
    name = "f_core_source_contribution_ratio"
    description = "核心源區基因貢獻度 - VWAP以下大戶買入量 vs 高價追買量的比值"
    required_columns = ["StockId", "Date", "raw_top10_low_buy", "raw_top10_high_buy"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        raw_val = df["raw_top10_low_buy"] / (df["raw_top10_high_buy"] + eps)
        
        out_series = g["raw_top10_low_buy"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
