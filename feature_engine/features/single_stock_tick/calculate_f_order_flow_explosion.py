import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureOrderFlowExplosion(BaseFeature):
    name = "f_flow_explosion"
    description = "內外盤力量爆發係數 (Order Flow Explosion Coefficient)"
    required_columns = ["StockId", "Date", "buy_vol", "sell_vol", "tick_count"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        buy_vol = df["buy_vol"].fillna(0)
        sell_vol = df["sell_vol"].fillna(0)
        
        imbalance = (buy_vol - sell_vol).abs() / (buy_vol + sell_vol + 1e-8)
        trade_count = df["tick_count"].fillna(1)
        
        explosion = imbalance * np.sqrt(trade_count)
        
        result = explosion.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
