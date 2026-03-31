import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureGtReputationCostSignaling(BaseFeature):
    name = "f_gt_reputation_cost_signaling"
    description = "Game Theory Module 4: Reputation cost signaling. Institutions flex buying power through 3-day decline."
    required_columns = ["StockId", "Date", "_ret_1d", "_inst_net", "_inst_net_change"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        # is_down = (_ret1d < 0).astype(int)
        is_down = (df["_ret_1d"] < 0).astype(int)
        
        # down_3d = (rolling sum of is_down over 3 days == 3).astype(int)
        # 3 consecutive down days
        down_3d = (g["_ret_1d"].rolling(3, min_periods=3).apply(lambda x: (x < 0).sum()) == 3).astype(int).values
        
        # is_net_positive = (_inst_net > 0).astype(int)
        is_net_positive = (df["_inst_net"] > 0).astype(int)
        
        # raw = down_3d * inst_net_change * is_net_positive
        # Note: _inst_net_change is already the diff(1) of _inst_net
        raw = down_3d * df["_inst_net_change"] * is_net_positive
        
        # zscore_rolling(20)
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
