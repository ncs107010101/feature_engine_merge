import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FCumulativeDoseThreshold(BaseFeature):
    name = "f_cumulative_dose_threshold"
    description = "累積劑量感染閥值 - 連續淨買入天數>=3天的券商其累計淨買入量佔全市場當日總成交量的比例"
    required_columns = ["StockId", "Date", "raw_infected_net_buy", "_total_net"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        raw_val = df["raw_infected_net_buy"] / (df["_total_net"] + eps)
        
        out_series = g["raw_infected_net_buy"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
