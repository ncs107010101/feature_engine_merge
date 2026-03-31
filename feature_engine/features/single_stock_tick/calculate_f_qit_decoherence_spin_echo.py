import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitDecoherenceSpinEcho(BaseFeature):
    name = "f_qit_decoherence_spin_echo"
    description = "自旋迴聲去相干測試。急跌後隨後3分鐘內止跌且超越原起跌點的發生次數。"
    
    required_columns = ["StockId", "Date", "raw_qit_decoherence_spin_echo"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_decoherence_spin_echo"]].copy()
        df.rename(columns={"raw_qit_decoherence_spin_echo": self.name}, inplace=True)
        return df
