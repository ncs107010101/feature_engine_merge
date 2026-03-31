import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureSmartMoneyCostStack(BaseFeature):
    name = "f_cost_stack"
    description = "主力成本堆疊高度 (Smart Money Cost Stack Height)"
    required_columns = ["StockId", "Date", "raw_top10_buy_vwap", "_atr_20", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        atr = df["_atr_20"]
        
        buy_price = df["raw_top10_buy_vwap"].replace(0, np.nan).fillna(df["收盤價"])
        stack_height = (df["收盤價"] - buy_price) / (atr + 1e-8)
        
        result = stack_height.fillna(0).groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, window=20, min_periods=1)
        )
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
