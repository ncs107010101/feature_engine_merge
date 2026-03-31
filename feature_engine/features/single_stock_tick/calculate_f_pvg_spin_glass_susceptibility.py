import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeaturePvgSpinGlassSusceptibility(BaseFeature):
    name = "f_pvg_spin_glass_susceptibility"
    description = (
        "自旋玻璃磁化率：100-tick bin 的訂單流方向視為自旋，"
        "計算自旋的自相關之和（磁化率 χ）× 平均自旋方向。"
    )
    required_columns = ["StockId", "Date", "raw_pvg_spin_glass_susceptibility"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_pvg_spin_glass_susceptibility"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
