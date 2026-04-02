import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import rolling_zscore


@register_feature
class FeatureDoubleThresholdNonlinearSurge(BaseFeature):
    name = "f_double_threshold_nonlinear_surge"
    description = "雙重閾值非線性躍變。同時跨越日均VWAP與開盤價兩閾值的主動買單量佔全日主動買單量的比例。"
    required_columns = ["StockId", "Date", "raw_cross_both_vol", "raw_total_active_buy"]
    data_combination = "cross_tick_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_cross_both_vol"] / (df["raw_total_active_buy"] + 1e-8)
        
        out = raw.groupby(df["StockId"]).transform(lambda x: rolling_zscore(x, 42))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
