import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeaturePriceRangeContractionDays(BaseFeature):
    name = "f_price_range_contraction_days"
    description = "Count of narrow-range days in last 10 days. Narrow = daily range < 60d avg range * 0.5."
    
    required_columns = ["StockId", "Date", "最高價", "最低價", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        
        # 1. Ensure time-series safety
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        # 2. Daily range ratio
        range_pct = (all_features['最高價'] - all_features['最低價']) / (all_features['收盤價'] + 1e-10)

        # 3. Time-Series operations
        # 60-day average range
        range_avg60 = range_pct.groupby(level="StockId").transform(
            lambda x: x.rolling(60, min_periods=10).mean()
        )

        # 4. Is today a narrow day?
        is_narrow = (range_pct < range_avg60 * 0.5).astype(float)
        is_narrow.loc[range_avg60.isna()] = 0.0

        # Count narrow days in last 10 days
        out_series = is_narrow.groupby(level="StockId").transform(
            lambda x: x.rolling(10, min_periods=1).sum()
        )

        # 5. Clean and format output
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
