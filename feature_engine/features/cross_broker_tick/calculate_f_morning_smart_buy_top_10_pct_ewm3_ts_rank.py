import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FMorningSmartBuyTop10pctEwm3TsRank(BaseFeature):
    name = "f_morning_smart_buy_top_10%_ewm3_ts_rank"
    description = "Profile feature: f_morning_smart_buy_top_10%_ewm3_ts_rank"
    required_columns = ["StockId", "Date", "r_morning_smart_buy_top_10_pct"]
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

        ewm = all_features.groupby(level="StockId")["r_morning_smart_buy_top_10_pct"].transform(lambda x: x.ewm(span=3).mean())
        out_series = ewm.groupby(level="StockId").transform(_ts_rank)
        
        # Post-processing
        out = pd.Series(out_series.values, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
