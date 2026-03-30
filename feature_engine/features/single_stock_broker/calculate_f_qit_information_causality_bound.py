import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitInformationCausalityBound(BaseFeature):
    name = "f_qit_information_causality_bound"
    description = "資訊因果界限溢出值。價格震幅比例除以參與券商數量。"
    
    required_columns = ["StockId", "Date", "raw_qit_information_causality_bound"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_information_causality_bound"]].copy()
        df.rename(columns={"raw_qit_information_causality_bound": self.name}, inplace=True)
        return df
