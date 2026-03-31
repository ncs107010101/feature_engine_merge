import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureSectorRotationPhase(BaseFeature):
    name = "f_phase_diff"
    description = "板塊輪動相位差 (Sector Rotation Phase Difference Approximation)"
    required_columns = ["StockId", "Date", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        ret_5d = df.groupby("StockId")["收盤價"].pct_change(5)
        ret_20d = df.groupby("StockId")["收盤價"].pct_change(20)
        
        diff = ret_5d - (ret_20d / 4)
        
        result = diff.fillna(0).groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
