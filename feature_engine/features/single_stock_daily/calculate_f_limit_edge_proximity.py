import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureLimitEdgeProximity(BaseFeature):
    name = "f_limit_edge"
    description = "漲跌停邊緣距離 (Limit Edge Proximity)"
    required_columns = ["StockId", "Date", "收盤價", "最高價", "最低價", "成交量(千股)"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        prev_close = df.groupby("StockId")["收盤價"].shift(1)
        
        # Taiwan stock price limit: 7% before 2015/6/1, 10% from 2015/6/1 onwards
        limit_pct = np.where(df["Date"].astype(int) < 20150601, 0.07, 0.10)
        ceil_pr = prev_close * (1 + limit_pct)
        floor_pr = prev_close * (1 - limit_pct)
        
        high = df["最高價"]
        low = df["最低價"]
        
        dist_to_ceil = (ceil_pr - high) / (ceil_pr + 1e-8)
        dist_to_floor = (low - floor_pr) / (floor_pr + 1e-8)
        
        limit_dist = np.minimum(dist_to_ceil, dist_to_floor)
        
        vol = df["成交量(千股)"]
        vol_ma20 = vol.groupby(df["StockId"]).rolling(window=20, min_periods=5).mean().reset_index(0, drop=True)
        vol_ratio = vol / (vol_ma20 + 1e-8)
        
        result = limit_dist * vol_ratio
        result = result.clip(0, 10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
