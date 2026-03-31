import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, rolling_slope

@register_feature
class FeatureNetBuyPersistenceSlope(BaseFeature):
    name = "f_net_buy_persistence_slope"
    description = "5-day slope of daily total net-buy volume, zscore(60)."
    
    required_columns = ["StockId", "Date", "raw_daily_net_buy"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        def _slope(s):
            return rolling_slope(s, window=5, min_periods=3)

        slope = all_features["raw_daily_net_buy"].groupby(level="StockId").transform(_slope)

        out_series = slope.groupby(level="StockId").transform(
            lambda x: zscore_rolling(x, window=60)
        )

        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
