import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FPrimarySecondaryInfectionDiv(BaseFeature):
    name = "f_primary_secondary_infection_div"
    description = "原發/次發感染背離 - Top5券商淨買入佔比與其餘券商買方參與率的差異"
    required_columns = ["StockId", "Date", "raw_top5_ratio", "raw_rest_participation"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        raw_val = df["raw_top5_ratio"] - df["raw_rest_participation"]
        
        out_series = g["raw_top5_ratio"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
