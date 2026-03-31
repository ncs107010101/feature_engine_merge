import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore


@register_feature
class FeatureBeInGroupFavoritismCross(BaseFeature):
    name = "f_be_in_group_favoritism_cross"
    description = "內團體偏愛交叉 - 主力買方一致性高且外部賣壓大"
    required_columns = ["StockId", "Date", "raw_top5_net_buy_nlargest", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # Get top5 brokers by cum_net (lookback 20)
        # top5_cv = std(top5_buy) / (mean(top5_buy) + 1e-5)
        # Using top5_net_buy_nlargest as proxy for top5 buy stats
        top5_buy_mean = df.groupby("StockId")["raw_top5_net_buy_nlargest"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        top5_buy_std = df.groupby("StockId")["raw_top5_net_buy_nlargest"].transform(
            lambda x: x.rolling(20, min_periods=1).std()
        )
        top5_cv = top5_buy_std / (top5_buy_mean + 1e-5)
        cohesion = 1.0 / (top5_cv + 1e-5)
        
        # outgroup_ns = rest_sell - rest_buy
        # Approximated as opposite of top5 net buy
        outgroup_ns = -df["raw_top5_net_buy_nlargest"]
        
        val = cohesion * outgroup_ns.clip(lower=0)
        
        out = ewm_then_zscore(val, ewm_span=5, z_window=20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
