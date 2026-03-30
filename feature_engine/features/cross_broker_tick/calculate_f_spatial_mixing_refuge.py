import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FSpatialMixingRefuge(BaseFeature):
    name = "f_spatial_mixing_refuge"
    description = "Top5 quiet_ratio - Rest quiet_ratio, zscore(42). 主力避開高波動時段，在尾盤平穩期低調建倉的超額吸籌比例。"
    required_columns = ["StockId", "Date", "raw_top5_quiet_ratio", "raw_rest_quiet_ratio"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        excess = all_features["raw_top5_quiet_ratio"] - all_features["raw_rest_quiet_ratio"]

        out = excess.groupby(level="StockId").transform(
            lambda x: zscore_rolling(x, window=42, min_periods=21)
        )

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return pd.DataFrame({
            self.name: out
        }).reset_index()
