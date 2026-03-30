import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FEma12(BaseFeature):
    name = "f_ema_12"
    description = "12-day Exponential Moving Average"
    required_columns = ["StockId", "Date", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        out_series = g["收盤價"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        
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
