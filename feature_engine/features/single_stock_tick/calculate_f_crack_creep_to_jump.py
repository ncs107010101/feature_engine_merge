import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureCrackCreepToJump(BaseFeature):
    name = "f_crack_creep_to_jump"
    description = (
        "潛變至斷裂臨界向量：向上/向下吃檔成交量的二階差分(加速度)差異。"
        "正值=向上吃檔在加速→看漲。"
    )
    required_columns = ["StockId", "Date", "raw_crack_creep_to_jump"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_crack_creep_to_jump"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
