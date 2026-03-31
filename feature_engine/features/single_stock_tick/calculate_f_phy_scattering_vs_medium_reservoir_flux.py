import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyScatteringVsMediumReservoirFlux(BaseFeature):
    name = "f_phy_scattering_vs_medium_reservoir_flux"
    description = "散射對介質水庫流量：報價總變化率×量×主動方向（×1000），衡量流動性摩擦力與方向性的乘積。"
    required_columns = ["StockId", "Date", "raw_phy_scattering_vs_medium_reservoir_flux"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_scattering_vs_medium_reservoir_flux"]]
            .copy()
            .rename(columns={"raw_phy_scattering_vs_medium_reservoir_flux": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
