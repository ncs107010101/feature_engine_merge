import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyNeutrinoEnergyDegradationRegeneration(BaseFeature):
    name = "f_phy_neutrino_energy_degradation_regeneration"
    description = "微中子能量衰退與再生：日內慣性×收益，衡量穩定累積能量後的方向性釋放強度。"
    required_columns = ["StockId", "Date", "raw_phy_neutrino_energy_degradation_regeneration"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_neutrino_energy_degradation_regeneration"]]
            .copy()
            .rename(columns={"raw_phy_neutrino_energy_degradation_regeneration": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
