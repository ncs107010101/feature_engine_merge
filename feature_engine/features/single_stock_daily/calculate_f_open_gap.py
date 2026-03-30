import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FOpenGap(BaseFeature):
    name = "f_open_gap"
    description = "Open gap"
    required_columns = ["StockId", "Date", "開盤價", "_prev_close"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        out_series = (df["開盤價"] - df["_prev_close"]) / (df["_prev_close"] + 1e-9)
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=-0.2, upper=0.2)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
