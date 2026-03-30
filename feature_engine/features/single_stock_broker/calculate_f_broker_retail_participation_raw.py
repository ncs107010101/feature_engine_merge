import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, mad_zscore, rolling_rank_pct

@register_feature
class FBrokerRetailParticipationRaw(BaseFeature):
    name = "f_broker_retail_participation_raw"
    description = "Broker feature: f_broker_retail_participation_raw"
    required_columns = ["StockId", "Date", "raw_retail_part"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = df["raw_retail_part"].copy()
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
