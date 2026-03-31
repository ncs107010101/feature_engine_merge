import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyDarkEmRepulsiveForce(BaseFeature):
    name = "f_phy_dark_em_repulsive_force"
    description = "暗電磁排斥力：散戶淨買率-大券商淨買率之差×日報酬絕對值的負值，衡量主散資金的相斥強度。"
    required_columns = ["StockId", "Date", "raw_phy_dark_em_repulsive_force"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_dark_em_repulsive_force"]]
            .copy()
            .rename(columns={"raw_phy_dark_em_repulsive_force": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
