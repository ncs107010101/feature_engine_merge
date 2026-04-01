import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, ts_rank_center

@register_feature
class FVpin(BaseFeature):
    name = "f_vpin"
    description = "Tick feature: f_vpin"
    required_columns = ["StockId", "Date", "raw_vpin"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["raw_vpin"].transform(lambda x: zscore_rolling(x, 60)).groupby(df["StockId"]).transform(lambda x: ewm_smooth(x, 5))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.clip(lower=-5, upper=5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
