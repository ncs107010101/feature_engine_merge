import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureSindyResidualAsymmetry(BaseFeature):
    name = "f_sindy_residual_asymmetry"
    description = (
        "系統識別殘差非對稱性：以 20-tick bins 的 EWMA 動量模型預測價格，"
        "計算預測殘差的偏態，偏態偏離高斯揭示隱蔽力量。"
    )
    required_columns = ["StockId", "Date", "raw_sindy_residual_asymmetry"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_sindy_residual_asymmetry"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
