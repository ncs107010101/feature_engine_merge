import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitIncompatibleObservablesProduct(BaseFeature):
    name = "f_qit_incompatible_observables_product"
    description = "不相容量子觀測積。位置(收盤價百分位距)與動量(尾盤主動買比率)的乘積。"
    
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_incompatible_observables_product"]].copy()
        df.rename(columns={"raw_qit_incompatible_observables_product": self.name}, inplace=True)
        return df
