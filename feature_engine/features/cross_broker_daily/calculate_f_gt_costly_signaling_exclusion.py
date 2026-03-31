import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtCostlySignalingExclusion(BaseFeature):
    name = "f_gt_costly_signaling_exclusion"
    description = "高昂信號排他鎖碼 - 主力高價買入鎖碼"
    required_columns = ["StockId", "Date", "raw_top5_net_buy_nlargest", "成交量(千股)", "成交金額(元)", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        daily_vwap = df["成交金額(元)"] / (df["成交量(千股)"] * 1000 + 1e-5)
        
        top3_avg_price = df["raw_top5_net_buy_nlargest"] * df["收盤價"]
        premium = top3_avg_price / (daily_vwap + 1e-5) - 1
        
        total_vol = df["成交量(千股)"] * 1000
        group_strength = np.abs(df["raw_top5_net_buy_nlargest"]) / (total_vol + 1e-5)
        
        val = pd.Series(np.where(premium > 0, np.maximum(0, premium) * group_strength, 0), index=df.index)
        
        out = zscore_rolling(val, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
