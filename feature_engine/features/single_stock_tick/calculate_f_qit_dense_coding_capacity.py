import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitDenseCodingCapacity(BaseFeature):
    name = "f_qit_dense_coding_capacity"
    description = "密集編碼資訊量。發生跨檔成交(跳檔)的成交量佔比。"
    
    required_columns = ["StockId", "Date", "raw_qit_dense_coding_capacity"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_dense_coding_capacity"]].copy()
        df.rename(columns={"raw_qit_dense_coding_capacity": self.name}, inplace=True)
        return df
