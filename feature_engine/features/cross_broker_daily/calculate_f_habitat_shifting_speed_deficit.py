import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FHabitatShiftingSpeedDeficit(BaseFeature):
    name = "f_habitat_shifting_speed_deficit"
    description = "棲地退移速度差 - 整體VWAP下移速度與Top10大戶買入VWAP下移速度的差"
    required_columns = ["StockId", "Date", "DailyVWAP", "raw_top10_buy_vwap"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        speed_habitat = df["DailyVWAP"].pct_change(fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
        speed_dispersal = df["raw_top10_buy_vwap"].pct_change(fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
        
        raw = speed_habitat - speed_dispersal
        
        out_series = g["DailyVWAP"].transform(lambda x: zscore_rolling(raw.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
