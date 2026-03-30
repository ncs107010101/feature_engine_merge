import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FInbreedingDepressionRisk(BaseFeature):
    name = "f_inbreeding_depression_risk"
    description = "近親繁殖風險 - Top5買入券商的5日Jaccard重合度乘以每日總買盤HHI"
    required_columns = ["StockId", "Date", "raw_top5_jaccard", "raw_broker_hhi"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        raw_val = df["raw_top5_jaccard"] * df["raw_broker_hhi"]
        
        out_series = g["raw_top5_jaccard"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
