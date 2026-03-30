import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyDislocationGlideClimbBias(BaseFeature):
    name = "f_phy_dislocation_glide_climb_bias"
    description = "錯位滑移攀移偏差：主動賣方與主動買方的量/幅效率比之對數差，衡量方向性偏斜。"
    required_columns = ["StockId", "Date", "raw_phy_dislocation_glide_climb_bias"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_dislocation_glide_climb_bias"]]
            .copy()
            .rename(columns={"raw_phy_dislocation_glide_climb_bias": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
