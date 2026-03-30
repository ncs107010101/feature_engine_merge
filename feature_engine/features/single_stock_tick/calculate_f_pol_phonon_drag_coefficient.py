import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeaturePolPhononDragCoefficient(BaseFeature):
    name = "f_pol_phonon_drag_coefficient"
    description = (
        "聲子拖曳係數：大單(≥90th pct)後面的小單跟隨率 × log(小單平均量/正常量) × 大單方向。"
        "正值=大單拉動小單追漲→動能持續。"
        "NOTE: 內部 preprocessing 使用 groupby.apply，因計算邏輯無法完全向量化。"
    )
    required_columns = ["StockId", "Date", "raw_pol_phonon_drag_coefficient"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_pol_phonon_drag_coefficient"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
