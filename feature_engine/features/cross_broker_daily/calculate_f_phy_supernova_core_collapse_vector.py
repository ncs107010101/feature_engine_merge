import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhySupernovaCoreCollapseVector(BaseFeature):
    name = "f_phy_supernova_core_collapse_vector"
    description = "超新星核心坍縮向量：散戶淨買-大券商淨買差×VWAP偏差的負值，衡量籌碼瞬間集中時的下行壓力。"
    required_columns = ["StockId", "Date", "raw_phy_supernova_core_collapse_vector"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_supernova_core_collapse_vector"]]
            .copy()
            .rename(columns={"raw_phy_supernova_core_collapse_vector": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
