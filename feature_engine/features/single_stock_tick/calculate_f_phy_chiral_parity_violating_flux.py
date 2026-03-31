import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyChiralParityViolatingFlux(BaseFeature):
    name = "f_phy_chiral_parity_violating_flux"
    description = "手徵宇稱違反流量：買賣價差變化率×量×主動方向，衡量手徵對稱性的破壞程度。"
    required_columns = ["StockId", "Date", "raw_phy_chiral_parity_violating_flux"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_chiral_parity_violating_flux"]]
            .copy()
            .rename(columns={"raw_phy_chiral_parity_violating_flux": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
