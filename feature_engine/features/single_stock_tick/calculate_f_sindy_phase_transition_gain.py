import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureSindyPhaseTransitionGain(BaseFeature):
    name = "f_sindy_phase_transition_gain"
    description = (
        "相變奇點資訊增益：比較日內前半和後半的 ΔP 分佈差異(KL散度)，"
        "結合方向。KL(Gaussian_first || Gaussian_second) × sign(μ2-μ1)。"
    )
    required_columns = ["StockId", "Date", "raw_sindy_phase_transition_gain"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_sindy_phase_transition_gain"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
