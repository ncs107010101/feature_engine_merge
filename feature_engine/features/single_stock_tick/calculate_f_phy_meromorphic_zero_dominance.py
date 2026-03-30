import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyMeromorphicZeroDominance(BaseFeature):
    name = "f_phy_meromorphic_zero_dominance"
    description = "亞純零點主導：中間價收益×時間間隔的積分，衡量報價靜止期的累積壓力。"
    required_columns = ["StockId", "Date", "raw_phy_meromorphic_zero_dominance"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_meromorphic_zero_dominance"]]
            .copy()
            .rename(columns={"raw_phy_meromorphic_zero_dominance": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
