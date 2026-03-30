import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureCollapseDirection(BaseFeature):
    name = "f_collapse_direction"
    description = "Tick feature f_collapse_direction"
    
    required_columns = ["StockId", "Date", "raw_collapse_direction"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        out_series = all_features['raw_collapse_direction'].groupby(level='StockId').transform(lambda x: x.ewm(span=3, min_periods=1).mean())

        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_result = pd.DataFrame({self.name: out}).reset_index()
        return final_result
