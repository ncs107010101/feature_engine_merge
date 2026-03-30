import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeaturePolEffectiveMassAsymmetry(BaseFeature):
    name = "f_pol_effective_mass_asymmetry"
    description = (
        "有效質量非對稱：Mass = Σvol / Σ|ΔP|，分別計算上漲/下跌 ticks，"
        "feature = (M_dn - M_up) / (M_dn + M_up)。正值=下跌慣性大→看跌。"
    )
    required_columns = ["StockId", "Date", "raw_pol_effective_mass_asymmetry"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_pol_effective_mass_asymmetry"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
