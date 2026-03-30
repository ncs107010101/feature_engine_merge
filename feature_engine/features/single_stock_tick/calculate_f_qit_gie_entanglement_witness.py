import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitGieEntanglementWitness(BaseFeature):
    name = "f_qit_gie_entanglement_witness"
    description = "重力誘導糾纏見證。1分鐘總成交量與VWAP偏離度的相關係數。"
    
    required_columns = ["StockId", "Date", "raw_qit_gie_entanglement_witness"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_gie_entanglement_witness"]].copy()
        df.rename(columns={"raw_qit_gie_entanglement_witness": self.name}, inplace=True)
        return df
