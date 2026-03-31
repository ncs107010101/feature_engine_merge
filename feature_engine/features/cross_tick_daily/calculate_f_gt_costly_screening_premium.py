import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtCostlyScreeningPremium(BaseFeature):
    name = "f_gt_costly_screening_premium"
    description = "is_costly × premium × (bin1_lb / total_lb), rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_bin50_bin1_large_buy", "raw_bin50_total_large_buy", "raw_screening_premium"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        df["raw_bin50_bin1_large_buy"] = df["raw_bin50_bin1_large_buy"].fillna(0)
        df["raw_bin50_total_large_buy"] = df["raw_bin50_total_large_buy"].fillna(0)
        df["raw_screening_premium"] = df["raw_screening_premium"].fillna(0)
        
        dominance = df["raw_bin50_bin1_large_buy"] / (df["raw_bin50_total_large_buy"] + 1e-5)
        is_costly = (df["raw_screening_premium"] > 0.01).astype(int)
        
        raw = is_costly * df["raw_screening_premium"] * dominance
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
