import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FVolumeSurprise(BaseFeature):
    name = "f_volume_surprise"
    description = "Volume surprise"
    required_columns = ["StockId", "Date", "成交量(千股)", "_vol_ma20", "_ret_1d"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        _vol_surge = df["成交量(千股)"] / (df["_vol_ma20"] + 1e-8)
        out_series = _vol_surge * np.sign(df["_ret_1d"])
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=-3, upper=3)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
