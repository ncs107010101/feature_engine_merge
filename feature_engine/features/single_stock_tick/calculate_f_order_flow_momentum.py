import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureOrderFlowMomentum(BaseFeature):
    name = "f_order_flow_momentum"
    description = "Tick feature f_order_flow_momentum"
    
    required_columns = ["StockId", "Date", "raw_order_flow_momentum"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        out_series = all_features['raw_order_flow_momentum'].groupby(level='StockId').transform(lambda x: zscore_rolling(x, window=20))

        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_result = pd.DataFrame({self.name: out}).reset_index()
        return final_result
