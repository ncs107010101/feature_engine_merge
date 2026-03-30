import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitHolevoCapacityUtilization(BaseFeature):
    name = "f_qit_holevo_capacity_utilization"
    description = "霍萊沃信道容量利用率。真實位移除以每分鐘K棒絕對位移總和。"
    
    required_columns = ["StockId", "Date", "raw_qit_holevo_capacity_utilization"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_holevo_capacity_utilization"]].copy()
        df.rename(columns={"raw_qit_holevo_capacity_utilization": self.name}, inplace=True)
        return df
