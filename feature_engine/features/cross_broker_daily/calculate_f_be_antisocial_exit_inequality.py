import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureBeAntisocialExitInequality(BaseFeature):
    name = "f_be_antisocial_exit_inequality"
    description = "反社會退出不平等 - 買方集中度高但參與者減少"
    required_columns = ["StockId", "Date", "raw_top5_net_buy_nlargest", "raw_buyer_count", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        top1_ratio = np.abs(df["raw_top5_net_buy_nlargest"]) / (np.abs(df["raw_top5_net_buy_nlargest"]) + 1e-5)
        df["top1_ratio"] = top1_ratio
        
        active_cnt = df["raw_buyer_count"].fillna(1)
        
        q90 = df.groupby("StockId")["top1_ratio"].transform(
            lambda x: x.rolling(42, min_periods=10).quantile(0.9)
        )
        
        inequality = (top1_ratio > q90).astype(int)
        
        avg_cnt = df.groupby("StockId")["raw_buyer_count"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        ).fillna(active_cnt)
        
        antisocial_exit = avg_cnt - active_cnt
        
        raw = inequality * antisocial_exit
        
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
