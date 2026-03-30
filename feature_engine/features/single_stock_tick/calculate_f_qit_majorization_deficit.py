import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitMajorizationDeficit(BaseFeature):
    name = "f_qit_majorization_deficit"
    description = "馬約化排序赤字。尾盤與早盤的主動買單洛倫茲曲線吉尼係數差值。"
    
    required_columns = ["StockId", "Date", "raw_qit_majorization_deficit"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_majorization_deficit"]].copy()
        df.rename(columns={"raw_qit_majorization_deficit": self.name}, inplace=True)
        return df
