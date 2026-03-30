import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, mad_zscore, rolling_rank_pct

@register_feature
class FBrokerConvictionBuyerRatio(BaseFeature):
    name = "f_broker_conviction_buyer_ratio"
    description = "Broker feature: f_broker_conviction_buyer_ratio"
    required_columns = ["StockId", "Date", "_conviction_buyer_ratio"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["_conviction_buyer_ratio"].transform(lambda x: ewm_smooth(x.fillna(0), 5))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
