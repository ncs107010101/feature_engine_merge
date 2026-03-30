import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import rolling_slope

@register_feature
class FeatureCloseToHighSlope(BaseFeature):
    name = "f_close_to_high_slope"
    description = "5-day slope of (High-Close)/(High-Low) ratio, smoothed by ewm(3)."
    
    required_columns = ["StockId", "Date", "最高價", "最低價", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        
        # 1. Ensure time-series safety
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        # 2. Close-to-high ratio
        rng = all_features['最高價'] - all_features['最低價']
        cth = (all_features['最高價'] - all_features['收盤價']) / (rng + 1e-10)
        cth = cth.clip(0, 1)

        # 3. Time-Series operations (rolling slope)
        # We need to apply rolling_slope within StockId groups
        def calc_slope(s):
            return rolling_slope(s, window=5, min_periods=3)
            
        slope = cth.groupby(level="StockId").transform(calc_slope)

        # EWM smoothing
        out_series = slope.groupby(level="StockId").transform(
            lambda x: x.ewm(span=3, min_periods=1).mean()
        )

        # 4. Clean and format output
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
