import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, mad_zscore, rolling_rank_pct

@register_feature
class FNewcomerNetMomentum(BaseFeature):
    name = "f_newcomer_net_momentum"
    description = "Broker feature: f_newcomer_net_momentum"
    required_columns = ["StockId", "Date", "_newcomer_net_momentum"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["_newcomer_net_momentum"].transform(lambda x: ewm_smooth(x.fillna(0), 5))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
