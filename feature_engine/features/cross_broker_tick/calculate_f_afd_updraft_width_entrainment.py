import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureAFDUpdraftWidthEntrainment(BaseFeature):
    name = "f_afd_updraft_width_entrainment"
    description = "Updraft width × positive price change"
    required_columns = ["StockId", "Date", "NB_BuyBrokers", "raw_delta_p"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        w_up = df["NB_BuyBrokers"]

        delta_p = df["raw_delta_p"].clip(lower=0)

        raw = w_up * delta_p

        out_series = raw.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, window=20, eps=1e-10)
        )

        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })