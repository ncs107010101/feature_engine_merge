import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureAlphaAcceleration(BaseFeature):
    name = "f_alpha_accel"
    description = "相對大盤阿爾法加速度 (Alpha Acceleration Relative to Index)"
    required_columns = ["StockId", "Date", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        ret = df.groupby("StockId")["收盤價"].pct_change()
        ret_diff1 = ret.groupby(df["StockId"]).diff()
        ret_diff2 = ret_diff1.groupby(df["StockId"]).shift(1)
        
        accel = ret_diff1 - ret_diff2
        
        result = accel.fillna(0).groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
