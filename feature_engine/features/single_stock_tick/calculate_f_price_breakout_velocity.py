import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeaturePriceBreakoutVelocity(BaseFeature):
    name = "f_breakout_velocity"
    description = "價格突破速度 (Price Breakout Velocity)"
    required_columns = ["StockId", "Date", "high_pr", "low_pr", "raw_breakout_time"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        price_range = df["high_pr"] - df["low_pr"]
        time_range = df["raw_breakout_time"].fillna(1)
        
        velocity = price_range / (np.sqrt(time_range) + 1)
        
        result = velocity.fillna(0).groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
