import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureVolumeAcceleration(BaseFeature):
    name = "f_vol_accel"
    description = "成交量加速度 (Volume Acceleration)"
    required_columns = ["StockId", "Date", "成交量(千股)"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        vol = df["成交量(千股)"].fillna(0)
        
        vol_shift1 = vol.groupby(df["StockId"]).shift(1)
        vol_shift2 = vol.groupby(df["StockId"]).shift(2)
        
        accel = vol - 2 * vol_shift1 + vol_shift2
        accel_normalized = accel / (vol_shift1.abs() + 1)
        
        result = accel_normalized.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
