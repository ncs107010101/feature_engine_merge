import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureVolatilityContractionTension(BaseFeature):
    name = "f_contraction_tension"
    description = "波動率收縮張力 (Volatility Contraction Tension)"
    required_columns = ["StockId", "Date", "收盤價", "最高價", "最低價", "成交量(千股)", "_atr_5", "_atr_60", "_vol_ma60"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        atr_5 = df["_atr_5"]
        atr_60 = df["_atr_60"]
        
        contraction = atr_60 / (atr_5 + 1e-8)
        
        vol = df["成交量(千股)"]
        vol_ma60 = df["_vol_ma60"]
        vol_ratio = vol / (vol_ma60 + 1e-8)
        
        tension = contraction * np.log1p(vol_ratio.clip(lower=0))
        result = tension.clip(0, 50).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
