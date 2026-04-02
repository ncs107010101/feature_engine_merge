import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQmCoarseGrainingLossBias(BaseFeature):
    name = "f_qm_coarse_graining_loss_bias"
    description = (
        "粗粒化資訊損失偏態：以 5 分鐘 bar 為解析度，計算向下 vs 向上時的"
        "tick 精細波動被「平均掉」的資訊損失非對稱性。"
    )
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_qm_coarse_graining_loss_bias"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
