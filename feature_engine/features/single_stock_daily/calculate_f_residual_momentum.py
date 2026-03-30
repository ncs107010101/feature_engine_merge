import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureResidualMomentum(BaseFeature):
    name = "f_residual_momentum"
    description = "殘餘動能係數 (Residual Momentum Coefficient)"
    required_columns = ["StockId", "Date", "收盤價", "最高價", "最低價", "_atr_20"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        atr = df["_atr_20"]
        
        ret_10d = df.groupby("StockId")["收盤價"].pct_change(10)
        ret_5d = df.groupby("StockId")["收盤價"].pct_change(5)
        ret_20d = df.groupby("StockId")["收盤價"].pct_change(20)
        
        residual = ret_10d - 0.5 * ret_5d - 0.5 * ret_20d
        
        result = residual / (atr / (df["收盤價"] + 1e-8) + 1e-8)
        result = result.clip(-5, 5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
