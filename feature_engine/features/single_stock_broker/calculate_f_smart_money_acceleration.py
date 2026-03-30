import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureSmartMoneyAcceleration(BaseFeature):
    name = "f_smart_money_accel"
    description = "主力連續買入加速度 (Smart Money Accumulation Acceleration)"
    required_columns = ["StockId", "Date", "_total_net"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        net_buy = df["_total_net"]
        net_buy_shift1 = net_buy.groupby(df["StockId"]).shift(1)
        net_buy_shift2 = net_buy.groupby(df["StockId"]).shift(2)
        
        accel = net_buy - 2 * net_buy_shift1 + net_buy_shift2
        accel_normalized = accel / (net_buy_shift1.abs() + 1)
        
        result = accel_normalized.fillna(0).groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
