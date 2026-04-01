import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureAFDEdgeWavePhaseLocking(BaseFeature):
    name = "f_afd_edge_wave_phase_locking"
    description = "Large vs retail momentum resonance"
    required_columns = ["StockId", "Date", "raw_top10_net_buy", "raw_retail_net_buy", "raw_tick_ret"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        w_top = df["raw_top10_net_buy"].groupby(df["StockId"]).transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        w_surf = df["raw_retail_net_buy"].groupby(df["StockId"]).transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

        dir_sign = np.sign(df["raw_tick_ret"])
        co_movement = w_top * w_surf

        raw = co_movement * dir_sign
        raw = np.log1p(np.abs(raw)) * np.sign(raw)
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, eps=1e-10))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: out})
