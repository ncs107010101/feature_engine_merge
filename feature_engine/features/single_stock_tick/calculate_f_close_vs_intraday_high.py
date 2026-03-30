import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, ts_rank_center

@register_feature
class FCloseVsIntradayHigh(BaseFeature):
    name = "f_close_vs_intraday_high"
    description = "Tick feature: f_close_vs_intraday_high"
    required_columns = ["StockId", "Date", "raw_close_vs_intraday_high"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["raw_close_vs_intraday_high"].transform(lambda x: ewm_smooth(x.fillna(0), 5))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
