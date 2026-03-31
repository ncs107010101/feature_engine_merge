import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FVolCv(BaseFeature):
    name = "f_vol_cv"
    description = "Volume CV"
    required_columns = ["StockId", "Date", "成交量(千股)", "_vol_ma20_orig"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        vol_std20 = g["成交量(千股)"].transform(lambda x: x.rolling(20).std())
        out_series = vol_std20 / (df["_vol_ma20_orig"] + 1e-9)
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=0, upper=10)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
