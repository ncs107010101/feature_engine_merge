import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureCriticalDensityRatio(BaseFeature):
    name = "f_density_ratio"
    description = "成交密度臨界比 (Critical Trading Density Ratio)"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        vol_daily = df["total_vol"].fillna(0)
        tick_daily = df["tick_count"].fillna(1)
        
        vol_ma = vol_daily.groupby(df["StockId"]).rolling(window=20, min_periods=5).mean().reset_index(0, drop=True)
        tick_ma = tick_daily.groupby(df["StockId"]).rolling(window=20, min_periods=5).mean().reset_index(0, drop=True)
        
        vol_peak_ratio = vol_daily / (vol_ma + 1e-8)
        tick_peak_ratio = tick_daily / (tick_ma + 1e-8)
        
        result = vol_peak_ratio / (tick_peak_ratio + 1e-8)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
