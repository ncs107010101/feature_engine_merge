import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyPhaseMatchingResonance(BaseFeature):
    name = "f_phy_phase_matching_resonance"
    description = "相位匹配共振：分鐘淨買單 autocorr(1) × 日淨買比率，衡量分鐘節律一致性與方向強度。"
    required_columns = ["StockId", "Date", "raw_phy_phase_matching_resonance"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_phase_matching_resonance"]]
            .copy()
            .rename(columns={"raw_phy_phase_matching_resonance": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
