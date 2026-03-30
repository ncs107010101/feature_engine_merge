import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureIntradayReversalDepth(BaseFeature):
    name = "f_intraday_reversal_depth"
    description = "Tick feature f_intraday_reversal_depth"
    
    required_columns = ["StockId", "Date", "raw_intraday_reversal_depth"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        out_series = all_features['raw_intraday_reversal_depth'].groupby(level='StockId').transform(lambda x: x.ewm(span=3, min_periods=1).mean())

        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_result = pd.DataFrame({self.name: out}).reset_index()
        return final_result
