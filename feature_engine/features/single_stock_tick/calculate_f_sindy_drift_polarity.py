import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureSindyDriftPolarity(BaseFeature):
    name = "f_sindy_drift_polarity"
    description = (
        "SINDy 漂移項極性：對 30-tick bins 的 ΔP/Δt ≈ α 做估計，"
        "提取 t-統計量 α/std_ret，衡量日內內生趨勢的強度與方向。"
    )
    required_columns = ["StockId", "Date", "raw_sindy_drift_polarity"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_sindy_drift_polarity"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
