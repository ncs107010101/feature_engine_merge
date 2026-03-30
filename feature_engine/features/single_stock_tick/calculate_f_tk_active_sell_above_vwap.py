import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, ts_rank_center

@register_feature
class FTkActiveSellAboveVwap(BaseFeature):
    name = "f_tk_active_sell_above_vwap"
    description = "Tick feature: f_tk_active_sell_above_vwap"
    required_columns = ["StockId", "Date", "raw_sell_above_vwap"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["raw_sell_above_vwap"].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
