import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyParityTransformationInvariant(BaseFeature):
    name = "f_phy_parity_transformation_invariant"
    description = "宇稱變換不變量：價差變化率×主動方向×量比率的強化版（×1000），對宇稱不對稱性的敏感度指標。"
    required_columns = ["StockId", "Date", "raw_phy_parity_transformation_invariant"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_parity_transformation_invariant"]]
            .copy()
            .rename(columns={"raw_phy_parity_transformation_invariant": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
