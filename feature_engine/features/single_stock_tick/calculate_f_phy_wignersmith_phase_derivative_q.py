import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyWignerSmithPhaseDerivativeQ(BaseFeature):
    name = "f_phy_wignersmith_phase_derivative_q"
    description = "Wigner-Smith 相位導數 Q：保留同向連續 tick 的累積動量，過濾反轉雜訊。"
    required_columns = ["StockId", "Date", "raw_phy_wignersmith_phase_derivative_q"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_wignersmith_phase_derivative_q"]]
            .copy()
            .rename(columns={"raw_phy_wignersmith_phase_derivative_q": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
