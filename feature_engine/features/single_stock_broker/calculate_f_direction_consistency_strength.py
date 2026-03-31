import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureDirectionConsistencyStrength(BaseFeature):
    name = "f_direction_consistency"
    description = "方向一致性強度 (Direction Consistency Strength)"
    required_columns = ["StockId", "Date", "_total_net"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        net_buy = df["_total_net"].fillna(0)
        net_buy_shift = net_buy.groupby(df["StockId"]).shift(1)
        
        # 方向一致性
        direction_agree = (net_buy * net_buy_shift) > 0
        
        # 淨買強度加成
        avg_abs_netbuy = net_buy.abs().groupby(df["StockId"]).rolling(window=20, min_periods=1).mean().reset_index(0, drop=True)
        strength_factor = 1 + (net_buy.abs() / (2 * avg_abs_netbuy + 1))
        
        combined = direction_agree.astype(float) * strength_factor
        result = combined.groupby(df["StockId"]).rolling(window=20, min_periods=1).mean().reset_index(0, drop=True)
        
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
