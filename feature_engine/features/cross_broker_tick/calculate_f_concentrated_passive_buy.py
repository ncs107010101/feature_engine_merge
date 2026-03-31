import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FConcentratedPassiveBuy(BaseFeature):
    name = "f_concentrated_passive_buy"
    description = "Profile feature: f_concentrated_passive_buy"
    required_columns = ["StockId", "Date", "f_concentrated_passive_buy"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        def _ts_rank(s, window=120):
            try:
                r = s.rolling(window, min_periods=5).rank(pct=True)
            except AttributeError:
                r = s.rolling(window, min_periods=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            return (r * 2) - 1

        def _ts_rolling_zscore(s, window=20):
            r = s.rolling(window, min_periods=5)
            return (s - r.mean()) / (r.std() + 1e-9)

        out_series = all_features["f_concentrated_passive_buy"]
        
        # Post-processing
        out = pd.Series(out_series.values, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
