import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeatureOverconfidenceSpillover(BaseFeature):
    name = "f_be_overconfidence_spillover"
    description = "Overconfidence Spillover (過度自信溢價): Top5 brokers from past 5 days with past_success + today_shock."
    required_columns = [
        "StockId", "Date",
        "top5_5d_netbuy", "past_success", "today_shock"
    ]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        condition = (df["past_success"] == 1) & (df["today_shock"] == 1)
        raw = np.where(condition, df["top5_5d_netbuy"], 0.0)
        raw = np.maximum(raw, 0)
        
        def rolling_zscore(x, window=20):
            mean = x.shift(1).rolling(window=window, min_periods=5).mean()
            std = x.shift(1).rolling(window=window, min_periods=5).std()
            return (x - mean) / (std + 1e-8)
        
        out = rolling_zscore(pd.Series(raw), 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
