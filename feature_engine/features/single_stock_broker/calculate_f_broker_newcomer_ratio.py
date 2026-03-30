import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import rolling_rank_pct

@register_feature
class FBrokerNewcomerRatio(BaseFeature):
    name = "f_broker_newcomer_ratio"
    description = "Top-10 net buy concentration (rolling rank normalized)"
    required_columns = ["StockId", "Date", "_top10_net_buy_concentration"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)

        out_series = g["_top10_net_buy_concentration"].transform(lambda x: rolling_rank_pct(x.fillna(0), window=20))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
