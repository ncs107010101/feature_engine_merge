import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtInGroupStealthCollusion(BaseFeature):
    name = "f_gt_in_group_stealth_collusion"
    description = "內團體隱蔽合謀 - 前5大券商與其他券商方向相反"
    required_columns = ["StockId", "Date", "raw_top5_net_buy_nlargest", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # Get top5 brokers by abs_net (lookback 20)
        # top5_net_buy = raw_top5_net_buy_nlargest
        # top5_nb = top5 NetBuy sum
        # rest_nb = rest NetBuy sum (approximated as negative of top5 for opposite direction)
        top5_nb = df["raw_top5_net_buy_nlargest"]
        rest_nb = -df["raw_top5_net_buy_nlargest"]
        
        collusion = ((top5_nb > 0) & (rest_nb < 0)).astype(int)
        intensity = np.where(collusion == 1, top5_nb / (np.abs(rest_nb) + 1e-5), 0)
        raw = collusion * intensity
        
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
