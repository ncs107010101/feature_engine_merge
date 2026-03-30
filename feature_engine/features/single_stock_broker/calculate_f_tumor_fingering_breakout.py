import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FTumorFingeringBreakout(BaseFeature):
    name = "f_tumor_fingering_breakout"
    description = "腫瘤指狀刺透力 - 最強買方的VWAP與其餘買方VWAP的乖離率"
    required_columns = ["StockId", "Date", "raw_top1_vwap", "raw_rest_vwap"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        raw_val = (df["raw_top1_vwap"] - df["raw_rest_vwap"]) / (df["raw_rest_vwap"] + eps)
        
        out_series = g["raw_top1_vwap"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
