import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FIntradayZscore(BaseFeature):
    name = "f_intraday_zscore"
    description = "Intraday zscore"
    required_columns = ["StockId", "Date", "_intra_ret"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        out_series = g["_intra_ret"].transform(lambda x: (x - x.rolling(100).mean()) / (x.rolling(100).std() + 1e-9))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=-5, upper=5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
