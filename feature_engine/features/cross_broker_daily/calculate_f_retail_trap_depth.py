import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureRetailTrapDepth(BaseFeature):
    name = "f_retail_trap"
    description = "散戶被套深度指數 (Retail Trap Depth Index)"
    required_columns = ["StockId", "Date", "raw_bot80_buy_vwap", "_atr_20"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        atr = df["_atr_20"]
        
        # Always use calculated VWAP from daily data (成交金額/成交量)
        # to maintain consistency with original implementation
        if "成交金額(元)" in df.columns and "成交量(千股)" in df.columns:
            vwap = df["成交金額(元)"] / (df["成交量(千股)"] * 1000 + 1e-8)
        else:
            vwap = df["收盤價"]
            
        trap_depth = (df["raw_bot80_buy_vwap"] - vwap).fillna(0) / (atr + 1e-8)
        
        result = trap_depth.fillna(0).groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, window=20, min_periods=1)
        )
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
