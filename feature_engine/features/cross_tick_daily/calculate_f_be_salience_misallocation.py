import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore


@register_feature
class FeatureBeSalienceMisallocation(BaseFeature):
    name = "f_be_salience_misallocation"
    description = "Salience / Volume Quality 的日內最大值, ewm_then_zscore(5, 20)"
    required_columns = ["StockId", "Date", "raw_salience_trap_max"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_salience_trap_max"]
        out = raw.groupby(df["StockId"]).transform(lambda x: ewm_then_zscore(x, 5, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
