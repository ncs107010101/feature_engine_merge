import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FCloseLocSmooth(BaseFeature):
    name = "f_close_loc_smooth"
    description = "Close location smooth"
    required_columns = ["StockId", "Date", "最高價", "最低價", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        hl_range = df["最高價"] - df["最低價"]
        _cl_loc = (df["收盤價"] - df["最低價"]) / (hl_range + 1e-9)
        out_series = _cl_loc.groupby(df["StockId"]).transform(lambda x: x.rolling(5).mean())
        
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
