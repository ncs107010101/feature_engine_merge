import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureLiquidityGapIntensity(BaseFeature):
    name = "f_liquidity_gap"
    description = "流動性缺口強度 (Liquidity Gap Intensity)"
    required_columns = ["StockId", "Date", "raw_tick_spread", "raw_tick_volatility"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        spread = df["raw_tick_spread"].fillna(0)
        volatility = df["raw_tick_volatility"].fillna(0)
        
        gap = spread * volatility
        
        result = gap.fillna(0).groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
