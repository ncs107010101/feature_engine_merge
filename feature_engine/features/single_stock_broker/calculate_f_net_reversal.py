import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, mad_zscore, rolling_rank_pct

@register_feature
class FNetReversal(BaseFeature):
    name = "f_net_reversal"
    description = "Broker feature: f_net_reversal"
    required_columns = ["StockId", "Date", "raw_net_reversal"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = -g["raw_net_reversal"].transform(lambda x: zscore_rolling(x, 5)).groupby(df["StockId"]).transform(lambda x: ewm_smooth(x, 10))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
