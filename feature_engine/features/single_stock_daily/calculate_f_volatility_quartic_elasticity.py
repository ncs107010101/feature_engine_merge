import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureVolatilityQuarticElasticity(BaseFeature):
    name = "f_vol_elasticity"
    description = "波動度四次方彈性 (Volatility Quartic Elasticity)"
    required_columns = ["StockId", "Date", "收盤價", "最高價", "最低價", "_atr_5", "_atr_20"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        atr_5 = df["_atr_5"]
        atr_20 = df["_atr_20"]
        
        ratio = atr_5 / (atr_20.replace(0, np.nan))
        ratio_safe = ratio.clip(lower=0.01, upper=10)
        
        # Non-linear expansion preserving directionality
        # Center around ratio=1 (neutral: short-term vol == long-term vol)
        # ratio > 1 → vol expansion (positive), ratio < 1 → vol contraction (negative)
        centered = ratio_safe - 1.0
        elastic = centered ** 3 + centered
        
        # Z-score normalize for stationarity
        result = elastic.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, window=60, min_periods=10)
        )
        result = result.clip(-5, 5)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
