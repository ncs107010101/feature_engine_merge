import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyInterstitialTopologicalStress(BaseFeature):
    name = "f_phy_interstitial_topological_stress"
    description = "間隙拓撲應力：Top5買方比率×(1-收盤位置) - Top5賣方比率×收盤位置，衡量主力在高低位的買賣壓力分佈。"
    required_columns = ["StockId", "Date", "raw_phy_interstitial_topological_stress"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_interstitial_topological_stress"]]
            .copy()
            .rename(columns={"raw_phy_interstitial_topological_stress": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
