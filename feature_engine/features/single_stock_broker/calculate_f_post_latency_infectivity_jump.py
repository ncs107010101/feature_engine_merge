import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FPostLatencyInfectivityJump(BaseFeature):
    name = "f_post_latency_infectivity_jump"
    description = "潛伏期後傳染跳躍 - 前5日大戶淨買入且價格橫盤後，當日主動買盤突然放大的程度"
    required_columns = ["StockId", "Date", "raw_sum5_top5_buy", "raw_price_range_5d", "raw_vol_jump"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        latency_score = df["raw_sum5_top5_buy"] / (df["raw_price_range_5d"] + 0.001)
        latency_score_prev = latency_score.shift(1).fillna(0)
        raw_val = latency_score_prev * df["raw_vol_jump"]
        
        df_indexed = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            "raw_val": raw_val
        }).set_index(["StockId", "Date"]).sort_index()
        
        df_indexed["zscore"] = df_indexed["raw_val"].groupby(level="StockId").transform(
            lambda x: zscore_rolling(x, 42)
        )
        
        df["zscore"] = df_indexed["zscore"].values
        
        out = df["zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
