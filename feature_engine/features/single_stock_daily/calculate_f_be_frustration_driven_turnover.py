import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FBeFrustrationDrivenTurnover(BaseFeature):
    name = "f_be_frustration_driven_turnover"
    description = "Behavioral Economics: Frustration-driven turnover. When price breaks 20-day high but closes lower, frustration triggers high turnover."
    required_columns = ["StockId", "Date", "最高價", "收盤價", "開盤價", "成交量(千股)", "流通在外股數(千股)"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        g = df.groupby(df["StockId"])
        
        prev_high_max = g["最高價"].transform(lambda x: x.rolling(20, min_periods=20).max().shift(1))
        
        high_expectation = (df["最高價"] > prev_high_max).astype(int)
        
        disappointment = (df["收盤價"] < df["開盤價"]).astype(int)
        
        turnover_rate = df["成交量(千股)"] / (df["流通在外股數(千股)"] + 1e-5)
        
        raw = high_expectation * disappointment * turnover_rate
        
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
