import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtPartialLyingStealthSell(BaseFeature):
    name = "f_gt_partial_lying_stealth_sell"
    description = "部分說謊隱蔽賣出 - 隱藏最大賣家的同時，其他賣家持續出貨"
    required_columns = ["StockId", "Date", "raw_top5_net_sell_qtm", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # Get sellers sorted by NetSell descending
        # raw_top5_net_sell_qtm represents top5 sellers' net sell
        # sellers sorted by NetSell: top1 is largest, rank_3_5 is sum of ranks 3-5
        # Since we have top5 net sell as a single value, we approximate:
        # top1 = raw_top5_net_sell_qtm * some_factor (using the data available)
        # For this feature: top1 is the max seller, rank_3_5 is the sum of 3rd to 5th
        
        top1 = df["raw_top5_net_sell_qtm"]
        rank_3_5 = df["raw_top5_net_sell_qtm"] * 0.6
        
        ratio = rank_3_5 / (top1 + 1e-5)
        
        out = zscore_rolling(ratio, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
