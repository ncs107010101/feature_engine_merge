import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdOptimalDragCoefficient(BaseFeature):
    name = "f_afd_optimal_drag_coefficient"
    description = "最佳表面阻力係數: 高斯核篩選 drag_ratio × slope, rolling_zscore(42)"
    required_columns = ["StockId", "Date", "raw_drag_ratio", "raw_tick_slope"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 42
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        drag_ratio = df["raw_drag_ratio"].values
        slope = df["raw_tick_slope"].values
        
        optimal = df.groupby("StockId")["raw_drag_ratio"].transform(
            lambda x: x.rolling(20, min_periods=5).median().fillna(x)
        ).values
        sigma = df.groupby("StockId")["raw_drag_ratio"].transform(
            lambda x: x.rolling(20, min_periods=5).std().fillna(1e-10)
        ).values + 1e-10
        kernel = np.exp(-((drag_ratio - optimal) ** 2) / (2 * sigma ** 2))
        raw = kernel * np.maximum(0, slope)
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
