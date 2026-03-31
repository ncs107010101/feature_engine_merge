import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import mad_zscore

@register_feature
class FBrokerCount(BaseFeature):
    name = "f_broker_hhi_concentration"
    description = "Active broker count (MAD z-score normalized)"
    required_columns = ["StockId", "Date", "_broker_count"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)

        out_series = g["_broker_count"].transform(lambda x: mad_zscore(x.fillna(0), window=20))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
