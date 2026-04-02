import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FTumorFingeringInstability(BaseFeature):
    name = "f_tumor_fingering_instability"
    description = "橫向指狀不穩定性 - 單一或少數大戶買在微觀最高價，刺穿上方阻力"
    required_columns = ["StockId", "Date", "raw_daily_max_price", "raw_daily_vwap", "raw_pioneer_buy", "raw_daily_total_buy"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        price_dev = (df["raw_daily_max_price"] - df["raw_daily_vwap"]) / (df["raw_daily_vwap"] + eps)
        pioneer_ratio = df["raw_pioneer_buy"] / (df["raw_daily_total_buy"] + eps)
        raw_val = price_dev * pioneer_ratio
        
        out_series = g["raw_daily_max_price"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
