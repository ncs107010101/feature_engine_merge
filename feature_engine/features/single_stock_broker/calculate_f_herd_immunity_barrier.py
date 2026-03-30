import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FHerdImmunityBarrier(BaseFeature):
    name = "f_herd_immunity_barrier"
    description = "群體免疫屏障效應 - 散戶平均買入成本與當前價格的乖離率"
    required_columns = ["StockId", "Date", "raw_retail_vwap", "DailyVWAP"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        retail_vwap_20 = g["raw_retail_vwap"].transform(lambda x: x.rolling(20, min_periods=1).mean())
        raw_val = (df["DailyVWAP"] - retail_vwap_20) / (retail_vwap_20 + eps)
        
        out_series = g["raw_retail_vwap"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
