import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdFrictionRiverbendExchange(BaseFeature):
    name = "f_afd_friction_riverbend_exchange"
    description = "河彎效應: spread不平衡 × 主動買量 × 正向ΔP, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_riverbend_val"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 20
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_riverbend_val"].values
        raw = pd.Series(raw)
        
        out = raw.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, self.ZSCORE_WINDOW, eps=1e-10)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
