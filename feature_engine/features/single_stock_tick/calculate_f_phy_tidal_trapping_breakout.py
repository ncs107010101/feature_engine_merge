import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyTidalTrappingBreakout(BaseFeature):
    name = "f_phy_tidal_trapping_breakout"
    description = "潮汐陷阱突破：VWAP 附近積聚的成交量陷阱乘以日內方向，衡量突破潛力。"
    required_columns = ["StockId", "Date", "raw_phy_tidal_trapping_breakout"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_tidal_trapping_breakout"]]
            .copy()
            .rename(columns={"raw_phy_tidal_trapping_breakout": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
