import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureLrtKuboResponsePolarization(BaseFeature):
    name = "f_lrt_kubo_response_polarization"
    description = (
        "庫柏響應極化度：識別大單衝擊(DealCount≥90th)，計算衝擊後5筆的價格響應係數，"
        "比較大買單和大賣單的響應差異。"
    )
    required_columns = ["StockId", "Date", "raw_lrt_kubo_response_polarization"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_lrt_kubo_response_polarization"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
