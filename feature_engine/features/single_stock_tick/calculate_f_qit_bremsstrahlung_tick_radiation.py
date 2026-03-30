import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitBremsstrahlungTickRadiation(BaseFeature):
    name = "f_qit_bremsstrahlung_tick_radiation"
    description = "制動輻射碎單強度。價格急跌減速瞬間，隨後30秒內的1張單(碎單)數量。"
    
    required_columns = ["StockId", "Date", "raw_qit_bremsstrahlung_tick_radiation"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_bremsstrahlung_tick_radiation"]].copy()
        df.rename(columns={"raw_qit_bremsstrahlung_tick_radiation": self.name}, inplace=True)
        return df
