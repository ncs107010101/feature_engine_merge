import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FShortChangeRoc(BaseFeature):
    name = "f_short_change_roc"
    description = "Short change ROC"
    required_columns = ["StockId", "Date", "_short_bal"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        out_series = g["_short_bal"].pct_change(3)
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=-2, upper=2)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
