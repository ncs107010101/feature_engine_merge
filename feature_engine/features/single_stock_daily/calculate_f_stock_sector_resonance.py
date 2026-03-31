import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureStockSectorResonance(BaseFeature):
    name = "f_sector_resonance"
    description = "個股-產業動能共振 (Stock-Sector Momentum Resonance) Approximation"
    required_columns = ["StockId", "Date", "收盤價", "成交量(千股)"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        ret_5d = df.groupby("StockId")["收盤價"].pct_change(5)
        
        vol = df["成交量(千股)"]
        vol_ma20 = vol.groupby(df["StockId"]).rolling(window=20, min_periods=5).mean().reset_index(0, drop=True)
        vol_ratio = vol / (vol_ma20 + 1e-8)
        
        result = ret_5d * vol_ratio * 100
        result = result.clip(-10, 10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
