import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, ts_rank_center

@register_feature
class FTradeSkew(BaseFeature):
    name = "f_trade_skew"
    description = "Tick feature: f_trade_skew"
    required_columns = ["StockId", "Date", "raw_trade_skew"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = -g["raw_trade_skew"].transform(lambda x: zscore_rolling(x, 60))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.clip(lower=-5, upper=5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
