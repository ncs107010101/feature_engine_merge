import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyCasimirVacuumPressureGradient(BaseFeature):
    name = "f_phy_casimir_vacuum_pressure_gradient"
    description = "卡西米爾真空壓力梯度：微量成交下的價格累積漂移，衡量無成交量時憑空出現的方向性壓力。"
    required_columns = ["StockId", "Date", "raw_phy_casimir_vacuum_pressure_gradient"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_casimir_vacuum_pressure_gradient"]]
            .copy()
            .rename(columns={"raw_phy_casimir_vacuum_pressure_gradient": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
