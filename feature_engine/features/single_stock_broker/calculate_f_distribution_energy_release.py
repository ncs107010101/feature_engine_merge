import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureDistributionEnergyRelease(BaseFeature):
    name = "f_distribution_energy"
    description = "倒貨能量釋放指數 (Distribution Energy Release Index)"
    required_columns = ["StockId", "Date", "_total_net"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        net_sell = (-df["_total_net"]).clip(lower=0)
        
        sell_days = (net_sell > 0).groupby(df["StockId"]).rolling(window=10, min_periods=1).sum().reset_index(0, drop=True) / 10
        
        avg_sell = net_sell.groupby(df["StockId"]).rolling(window=60, min_periods=20).mean().reset_index(0, drop=True)
        sell_intensity = np.log1p(net_sell / (avg_sell + 1e-8))
        
        result_raw = sell_days * sell_intensity
        
        result = result_raw.fillna(0).groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
