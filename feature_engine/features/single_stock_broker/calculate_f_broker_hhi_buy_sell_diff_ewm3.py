import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, mad_zscore, rolling_rank_pct

@register_feature
class FBrokerHhiBuySellDiffEwm3(BaseFeature):
    name = "f_broker_hhi_buy_sell_diff_ewm3"
    description = "Broker feature: f_broker_hhi_buy_sell_diff_ewm3"
    required_columns = ["StockId", "Date", "raw_hhi_diff"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["raw_hhi_diff"].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
