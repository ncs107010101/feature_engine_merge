import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureMhdBeltramiAlignment(BaseFeature):
    name = "f_mhd_beltrami_alignment"
    description = (
        "貝特拉米無力場對齊度：∇×B=λB 時電流與磁場平行無耗散。"
        "B=50-tick bin 內價格趨勢，J=大單淨流加速度，取 cosine similarity。"
    )
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_mhd_beltrami_alignment"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
