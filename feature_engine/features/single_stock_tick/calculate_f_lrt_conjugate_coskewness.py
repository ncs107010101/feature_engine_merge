import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureLrtConjugateCoskewness(BaseFeature):
    name = "f_lrt_conjugate_coskewness"
    description = (
        "共軛變數不確定性偏態：計算 50-tick bin 內 (成交量, VWAP偏離度) 的 Co-Skewness。"
        "E[(V-μV)²(P-μP)] / (σV² × σP)。"
    )
    required_columns = ["StockId", "Date", "raw_lrt_conjugate_coskewness"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_lrt_conjugate_coskewness"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
