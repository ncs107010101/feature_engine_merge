import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureAFDJetExitRadiation(BaseFeature):
    name = "f_afd_jet_exit_radiation"
    description = "Large investor net buy × price acceleration"
    required_columns = ["StockId", "Date", "raw_top5_net_buy", "raw_tick_ret"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        j = np.maximum(0, df["raw_top5_net_buy"])
        a_price = df.groupby("StockId")["raw_tick_ret"].transform(lambda x: x.diff().fillna(0))

        raw = j * a_price
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, eps=1e-10))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: out})
