import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureBePriorBiasNeglect(BaseFeature):
    name = "f_be_prior_bias_neglect"
    description = "前日跳空低開 × 反彈區間的小單賣出額 / 總成交量, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_prior_pessimism", "raw_ignorant_selling", "raw_total_vol_50bin"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_prior_pessimism"] * df["raw_ignorant_selling"] / (df["raw_total_vol_50bin"] + 1e-8)
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })