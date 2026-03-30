import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FHighLowRangeExpansion(BaseFeature):
    name = "f_high_low_range_expansion"
    description = "High low range expansion"
    required_columns = ["StockId", "Date", "_ret_1d"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        _mad_5d = g["_ret_1d"].transform(lambda x: x.abs().rolling(5, min_periods=3).median())
        _mad_20d = g["_ret_1d"].transform(lambda x: x.abs().rolling(20, min_periods=10).median())
        out_series = pd.Series(_mad_5d / (_mad_20d + 1e-8))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=0, upper=3)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
