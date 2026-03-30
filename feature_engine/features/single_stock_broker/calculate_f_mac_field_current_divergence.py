import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureMacFieldCurrentDivergence(BaseFeature):
    name = "f_mac_field_current_divergence"
    description = "f_mac_field_current_divergence physics feature"
    
    required_columns = ["StockId", "Date", "raw_mac_field_current_divergence"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # 1. 確保按照時間排序 (Financial safety)
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        out = all_features["raw_mac_field_current_divergence"]
        
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
