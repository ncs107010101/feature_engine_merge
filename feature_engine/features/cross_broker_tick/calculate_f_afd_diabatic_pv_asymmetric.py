import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureAFDDiabaticPVAsymmetric(BaseFeature):
    name = "f_afd_diabatic_pv_asymmetric"
    description = "Broker buy 2nd-order diff × price return"
    required_columns = ["StockId", "Date", "raw_top10_net_buy", "raw_retail_net_buy", "raw_tick_ret"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        j_buy = df.groupby("StockId")["raw_top10_net_buy"].transform(lambda x: x.diff().diff().fillna(0))
        j_sell = df.groupby("StockId")["raw_retail_net_buy"].transform(lambda x: x.diff().diff().fillna(0))

        raw = np.maximum(0, j_buy) * df["raw_tick_ret"] - np.minimum(0, j_sell) * df["raw_tick_ret"]
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, eps=1e-10))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: out})
