import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureBeMoodCongruentAction(BaseFeature):
    name = "f_be_mood_congruent_action"
    description = "前日重挫跳空 × 上漲區間的小單賣出額, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_neg_mood", "raw_mood_sell"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_neg_mood"] * df["raw_mood_sell"]
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })