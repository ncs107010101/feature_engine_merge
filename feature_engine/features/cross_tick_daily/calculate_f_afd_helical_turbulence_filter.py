import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdHelicalTurbulenceFilter(BaseFeature):
    name = "f_afd_helical_turbulence_filter"
    description = "高螺旋度壓制耗散: 大單買量/小單買量 × 正向價格變動, log1p, rolling_zscore(42)"
    required_columns = ["StockId", "Date", "q85_large_buy", "q85_small_buy", "close_pr", "open_pr"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 42
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        signal = df["q85_large_buy"].values.astype(np.float64)
        noise = df["q85_small_buy"].values.astype(np.float64) + 1.0
        delta_p = np.maximum(0, df["close_pr"].values - df["open_pr"].values)
        
        raw = (signal / noise) * delta_p
        raw = np.log1p(raw)
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
