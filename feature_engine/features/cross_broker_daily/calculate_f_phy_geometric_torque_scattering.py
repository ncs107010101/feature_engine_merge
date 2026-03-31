import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyGeometricTorqueScattering(BaseFeature):
    name = "f_phy_geometric_torque_scattering"
    description = "幾何扭矩散射：Top5淨買率×近20日高低點距離引力，衡量主力在技術關鍵位附近的拉升力。"
    required_columns = ["StockId", "Date", "raw_phy_geometric_torque_scattering"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_geometric_torque_scattering"]]
            .copy()
            .rename(columns={"raw_phy_geometric_torque_scattering": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
