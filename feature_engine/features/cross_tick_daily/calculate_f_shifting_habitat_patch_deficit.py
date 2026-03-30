import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import rolling_zscore


@register_feature
class FeatureShiftingHabitatPatchDeficit(BaseFeature):
    name = "f_shifting_habitat_patch_deficit"
    description = "斑塊存活赤字。價格重心推升過快，但最大成交量區間極度單薄。推升速度 / (日內mode_price ±0.5%成交量佔比 + ε)。"
    required_columns = ["StockId", "Date", "raw_push_speed", "raw_patch_thickness"]
    data_combination = "cross_tick_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_push_speed"] / (df["raw_patch_thickness"] + 1e-8)
        
        out = raw.groupby(df["StockId"]).transform(lambda x: rolling_zscore(x, 42))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
