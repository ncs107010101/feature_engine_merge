import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitAthermalFreeEnergy(BaseFeature):
    name = "f_qit_athermal_free_energy"
    description = "微結構非熱力學自由能。買超券商淨買入分佈與均勻分佈的KL散度(熵)。"
    
    required_columns = ["StockId", "Date", "raw_qit_athermal_free_energy"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_athermal_free_energy"]].copy()
        df.rename(columns={"raw_qit_athermal_free_energy": self.name}, inplace=True)
        return df
