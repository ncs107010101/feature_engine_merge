import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureDptNonreciprocalInteraction(BaseFeature):
    name = "f_dpt_nonreciprocal_interaction"
    description = "f_dpt_nonreciprocal_interaction physics feature"
    
    required_columns = ["StockId", "Date", "raw_dpt_nonreciprocal_interaction"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # 1. 確保按照時間排序 (Financial safety)
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        out = all_features["raw_dpt_nonreciprocal_interaction"]
        
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
