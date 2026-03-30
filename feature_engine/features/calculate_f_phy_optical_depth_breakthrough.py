import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyOpticalDepthBreakthrough(BaseFeature):
    name = "f_phy_optical_depth_breakthrough"
    description = "光學深度突破：歷史累積量比率加權的收益率，衡量穿透已建倉籌碼的代價。"
    required_columns = ["StockId", "Date", "raw_phy_optical_depth_breakthrough"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_optical_depth_breakthrough"]]
            .copy()
            .rename(columns={"raw_phy_optical_depth_breakthrough": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
