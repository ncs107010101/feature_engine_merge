import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, mad_zscore, rolling_rank_pct

@register_feature
class FBrokerPolarizationIndex(BaseFeature):
    name = "f_broker_polarization_index"
    description = "Broker feature: f_broker_polarization_index"
    required_columns = ["StockId", "Date", "_polarization_index"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["_polarization_index"].transform(lambda x: ewm_smooth(x.fillna(0), 5))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
