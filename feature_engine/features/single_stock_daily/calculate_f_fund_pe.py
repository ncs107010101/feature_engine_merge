import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FFundPe(BaseFeature):
    name = "f_fund_pe"
    description = "Fund PE"
    required_columns = ["StockId", "Date", "本益比(TEJ)"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        out_series = df["本益比(TEJ)"].copy()
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
