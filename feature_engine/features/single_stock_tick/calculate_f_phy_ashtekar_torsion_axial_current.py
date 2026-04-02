import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyAshtekarTorsionAxialCurrent(BaseFeature):
    name = "f_phy_ashtekar_torsion_axial_current"
    description = "Ashtekar 扭量軸向電流：買賣報價夾角的偏斜量加權積分，衡量撮合傾向的不對稱性。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_ashtekar_torsion_axial_current"]]
            .copy()
            .rename(columns={"raw_phy_ashtekar_torsion_axial_current": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
