import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FTurnoverRate(BaseFeature):
    name = "f_turnover_rate"
    description = "Turnover rate"
    required_columns = ["StockId", "Date", "_turnover_rate"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        out_series = df["_turnover_rate"]
        
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
