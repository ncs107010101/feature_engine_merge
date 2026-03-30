import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitOtocScramblingRate(BaseFeature):
    name = "f_qit_otoc_scrambling_rate"
    description = "OTOC 資訊擾動率。1分鐘報酬率的自我相關係數衰減。"
    
    required_columns = ["StockId", "Date", "raw_qit_otoc_scrambling_rate"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_otoc_scrambling_rate"]].copy()
        df.rename(columns={"raw_qit_otoc_scrambling_rate": self.name}, inplace=True)
        return df
