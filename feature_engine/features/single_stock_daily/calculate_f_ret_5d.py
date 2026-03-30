import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FRet5d(BaseFeature):
    name = "f_ret_5d"
    description = "5-day return"
    required_columns = ["StockId", "Date", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        out_series = g["收盤價"].pct_change(5)
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=-0.5, upper=0.5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
