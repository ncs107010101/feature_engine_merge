import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureNetBuyEnergyIntegral(BaseFeature):
    name = "f_energy_integral"
    description = "淨買能量積分 (Net Buying Energy Integral)"
    required_columns = ["StockId", "Date", "_total_net"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        net_buy = df["_total_net"].fillna(0)
        
        def _weighted_energy(s):
            x = s.values
            weights = np.array([(i + 1) ** 0.5 for i in range(len(x))])
            return np.sum(x * weights) / (np.sqrt(len(x)) + 1e-8)
            
        energy = net_buy.groupby(df["StockId"]).rolling(window=20, min_periods=1).apply(_weighted_energy, raw=False).reset_index(0, drop=True)
        
        result = energy.fillna(0).groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
