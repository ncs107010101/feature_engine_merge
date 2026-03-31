import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, ts_rank_center

@register_feature
class FTkFailedSurgeTrap(BaseFeature):
    name = "f_tk_failed_surge_trap"
    description = "Tick feature: f_tk_failed_surge_trap"
    required_columns = ["StockId", "Date", "raw_failed_surge"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["raw_failed_surge"].transform(lambda x: x.ewm(span=5, min_periods=1).mean()).groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20)).groupby(df["StockId"]).transform(lambda x: x.ewm(span=10).mean())
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
