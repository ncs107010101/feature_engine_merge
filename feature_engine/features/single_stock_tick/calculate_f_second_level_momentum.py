import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureSecondLevelMomentum(BaseFeature):
    name = "f_momentum_continuity"
    description = "秒級動量連續性 (Second-Level Momentum Continuity)"
    required_columns = ["StockId", "Date", "raw_momentum_continuity"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        continuity = df["raw_momentum_continuity"].fillna(0.5)
        
        result = continuity.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
