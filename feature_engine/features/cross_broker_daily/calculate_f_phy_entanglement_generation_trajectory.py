import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyEntanglementGenerationTrajectory(BaseFeature):
    name = "f_phy_entanglement_generation_trajectory"
    description = "糾纏生成軌跡：Top5淨買率×報酬的20日相關×Top5淨買率，衡量主力資金與行情方向的協同性。"
    required_columns = ["StockId", "Date", "raw_phy_entanglement_generation_trajectory"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_entanglement_generation_trajectory"]]
            .copy()
            .rename(columns={"raw_phy_entanglement_generation_trajectory": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
