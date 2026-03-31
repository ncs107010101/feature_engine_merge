import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureGeoLyapunovDriftVector(BaseFeature):
    name = "f_geo_lyapunov_drift_vector"
    description = (
        "黎曼流形 Lyapunov 漂移向量：對 10-tick bins 的 ΔP 序列計算二階差分 (加速度)的 EWMA，"
        "標準化後衡量動量在流形上的持續漂移方向。"
    )
    required_columns = ["StockId", "Date", "raw_geo_lyapunov_drift_vector"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_geo_lyapunov_drift_vector"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
