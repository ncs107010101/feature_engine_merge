import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, ts_rank_center

@register_feature
class FDbTrendRsquared(BaseFeature):
    name = "f_db_trend_rsquared"
    description = "Tick feature: f_db_trend_rsquared"
    required_columns = ["StockId", "Date", "raw_db_trend_r2"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["raw_db_trend_r2"].transform(lambda x: zscore_rolling(x.ewm(span=10).mean(), 20))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
