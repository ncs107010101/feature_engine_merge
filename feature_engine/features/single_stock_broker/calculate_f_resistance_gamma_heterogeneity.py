import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FResistanceGammaHeterogeneity(BaseFeature):
    name = "f_resistance_gamma_heterogeneity"
    description = "抗性Gamma異質性 - 賣出券商的HHI除以賣出券商家數"
    required_columns = ["StockId", "Date", "raw_broker_hhi", "raw_seller_count"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        raw_val = df["raw_broker_hhi"] / (df["raw_seller_count"] + eps)
        
        out_series = g["raw_broker_hhi"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
