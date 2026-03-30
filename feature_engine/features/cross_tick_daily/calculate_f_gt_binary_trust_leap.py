import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtBinaryTrustLeap(BaseFeature):
    name = "f_gt_binary_trust_leap"
    description = "近20日漲幅>15% × (開盤首個Bin的大單買入/20日均值), rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_high_prior", "raw_bin1_large_buy"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        df["hist_mean"] = df.groupby("StockId")["raw_bin1_large_buy"].transform(
            lambda x: x.rolling(20, min_periods=5).mean().shift(1)
        )
        df["hist_mean"] = df["hist_mean"].fillna(df["raw_bin1_large_buy"])
        
        raw = df["raw_high_prior"] * df["raw_bin1_large_buy"] / (df["hist_mean"] + 1e-8)
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })