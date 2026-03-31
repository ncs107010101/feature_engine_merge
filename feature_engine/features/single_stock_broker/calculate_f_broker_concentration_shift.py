import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureBrokerConcentrationShift(BaseFeature):
    name = "f_broker_concentration_shift"
    description = "Change in broker HHI (Herfindahl) concentration from 5 days ago, zscore(20)."
    
    required_columns = ["StockId", "Date", "raw_broker_hhi"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        shift5 = all_features["raw_broker_hhi"].groupby(level="StockId").shift(5)
        hhi_shift = all_features["raw_broker_hhi"] - shift5

        out_series = hhi_shift.groupby(level="StockId").transform(
            lambda x: zscore_rolling(x, window=20)
        )

        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
