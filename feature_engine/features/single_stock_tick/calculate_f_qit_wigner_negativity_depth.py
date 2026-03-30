import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitWignerNegativityDepth(BaseFeature):
    name = "f_qit_wigner_negativity_depth"
    description = "維格納函數負值深度。相空間中 Wigner Function 的負值區域代表純粹的非古典特徵。異常量價背離區間。"
    
    required_columns = ["StockId", "Date", "raw_qit_wigner_negativity_depth"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_wigner_negativity_depth"]].copy()
        df.rename(columns={"raw_qit_wigner_negativity_depth": self.name}, inplace=True)
        return df
