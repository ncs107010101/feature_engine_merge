import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyDcdwCollinearAcceleration(BaseFeature):
    name = "f_phy_dcdw_collinear_acceleration"
    description = "DCDW 共線加速度：只保留量與價同向的共線分量，衡量主動方向性資金的加速驅動力。"
    required_columns = ["StockId", "Date", "raw_phy_dcdw_collinear_acceleration"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_dcdw_collinear_acceleration"]]
            .copy()
            .rename(columns={"raw_phy_dcdw_collinear_acceleration": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
