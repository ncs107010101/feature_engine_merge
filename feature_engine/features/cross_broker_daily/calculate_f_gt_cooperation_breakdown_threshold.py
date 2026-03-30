import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtCooperationBreakdownThreshold(BaseFeature):
    name = "f_gt_cooperation_breakdown_threshold"
    description = "合作崩潰閾值 - 買方券商數激增但主力券商叛離"
    required_columns = ["StockId", "Date", "raw_buyer_count", "raw_top5_net_buy_nlargest", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # buy_group_size = count of brokers with NetBuy > 0
        buy_group_size = df["raw_buyer_count"]
        
        max20 = df.groupby("StockId")["raw_buyer_count"].transform(
            lambda x: x.rolling(20, min_periods=1).max().shift(1)
        )
        beyond_threshold = (buy_group_size > max20).astype(int)
        
        # Get top5 brokers by cum_net (lookback 20)
        # top5_net_buy = raw_top5_net_buy_nlargest
        # top5_defection = max(0, -(top5 NetBuy sum))
        top5_defection = df["raw_top5_net_buy_nlargest"].apply(lambda x: max(0, -x))
        
        raw = beyond_threshold * top5_defection
        
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
