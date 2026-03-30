import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtImmediateDisclosurePanic(BaseFeature):
    name = "f_gt_immediate_disclosure_panic"
    description = "Game Theory Module 5: Immediate Disclosure Panic - Gap down ≥ 2% + heavy active selling in first 500 ticks. Positive → Violent information dump → Extreme LOW return."
    required_columns = ["StockId", "Date", "開盤價", "收盤價", "報酬率", "raw_first500_sell_ratio"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        df['ref_price'] = df['收盤價'] / (df['報酬率'] / 100 + 1)
        df['overnight_return'] = df['開盤價'] / df['ref_price'] - 1
        df['gap_down'] = (df['overnight_return'] < -0.02).astype(int)
        
        df['raw'] = df['gap_down'] * df['raw_first500_sell_ratio']
        result = df.groupby('StockId')['raw'].transform(lambda x: zscore_rolling(x, 20))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: result
        })