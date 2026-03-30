import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureGtUnravelingSkepticismFailure(BaseFeature):
    name = "f_gt_unraveling_skepticism_failure"
    description = "Game Theory Module 3: Unraveling skepticism failure. When retail increases holdings despite bad revenue and falling prices."
    required_columns = ["StockId", "Date", "_rev_yoy", "_ret_5d", "_retail_count"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        is_bad = ((df["_rev_yoy"] <= 0) & (df["_ret_5d"] <= 0)).astype(int)
        retail_growth_5 = g["_retail_count"].pct_change(5)
        raw = is_bad * retail_growth_5
        out = raw.ewm(span=3, min_periods=1).mean()
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
