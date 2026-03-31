import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureMhdMagneticHelicity(BaseFeature):
    name = "f_mhd_magnetic_helicity"
    description = (
        "訂單流磁螺旋度：A=cumsum(NetActiveFlow)，B=tanh(ΔP×10)。"
        "H_daily = Σ A(t)·B(t) / (N × max|A|)，衡量動量累積與價格方向的螺旋纏繞程度。"
    )
    required_columns = ["StockId", "Date", "raw_mhd_magnetic_helicity"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_mhd_magnetic_helicity"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
