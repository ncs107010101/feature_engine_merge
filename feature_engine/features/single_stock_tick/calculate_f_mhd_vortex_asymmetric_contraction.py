import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureMhdVortexAsymmetricContraction(BaseFeature):
    name = "f_mhd_vortex_asymmetric_contraction"
    description = (
        "渦旋相空間不對稱收縮：比較上漲/下跌 tick 在 (ΔP/DealCount, DealCount)"
        "相空間中協方差矩陣行列式的對數比值。"
    )
    required_columns = ["StockId", "Date", "raw_mhd_vortex_asymmetric_contraction"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_mhd_vortex_asymmetric_contraction"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
