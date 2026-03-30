import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FStoichiometricNutrientImbalance(BaseFeature):
    name = "f_stoichiometric_nutrient_imbalance"
    description = "資本-籌碼化學計量失衡 - Top10大戶現金流淨買入與Bot80%小戶淨賣出股數的對數比"
    required_columns = ["StockId", "Date", "raw_top10_net_value", "raw_bot80_net_sell_qtm"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        raw_val = np.log((df["raw_top10_net_value"] + 1) / (df["raw_bot80_net_sell_qtm"] + 1))
        
        out_series = g["raw_top10_net_value"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
