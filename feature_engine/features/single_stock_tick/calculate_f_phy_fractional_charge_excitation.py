import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyFractionalChargeExcitation(BaseFeature):
    name = "f_phy_fractional_charge_excitation"
    description = "分數電荷激發：相對停留時長×量比率×方向，衡量同價停留的方向性積累密度（price-scale invariant）。"
    required_columns = ["StockId", "Date", "raw_phy_fractional_charge_excitation"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_fractional_charge_excitation"]]
            .copy()
            .rename(columns={"raw_phy_fractional_charge_excitation": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
