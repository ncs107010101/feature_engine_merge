import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQmDensityOffdiagRelaxation(BaseFeature):
    name = "f_qm_density_offdiag_relaxation"
    description = (
        "密度矩陣非對角項退相干：計算大/小訂單流之間的交叉相關衰減率，"
        "× lag-1 相關的符號，衡量市場自組織退相干速率。"
    )
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_qm_density_offdiag_relaxation"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
