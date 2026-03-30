import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureEthConservativeWorkDivergence(BaseFeature):
    name = "f_eth_conservative_work_divergence"
    description = "f_eth_conservative_work_divergence physics feature"
    
    required_columns = ["StockId", "Date", "raw_eth_conservative_work_divergence"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # 1. 確保按照時間排序 (Financial safety)
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        out = all_features["raw_eth_conservative_work_divergence"]
        
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
