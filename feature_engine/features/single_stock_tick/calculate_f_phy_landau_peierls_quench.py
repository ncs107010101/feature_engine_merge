import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyLandauPeierlsQuench(BaseFeature):
    name = "f_phy_landau_peierls_quench"
    description = "Landau-Peierls 驟冷：近50筆的反轉頻率×量比率×方向，衡量高頻震盪後的定向淨能量。"
    required_columns = ["StockId", "Date", "raw_phy_landau_peierls_quench"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_landau_peierls_quench"]]
            .copy()
            .rename(columns={"raw_phy_landau_peierls_quench": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
