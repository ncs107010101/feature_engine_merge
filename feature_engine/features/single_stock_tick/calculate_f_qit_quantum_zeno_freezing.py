import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitQuantumZenoFreezing(BaseFeature):
    name = "f_qit_quantum_zeno_freezing"
    description = "量子芝諾凍結指數。連續密集測量凍結價格機率。"
    
    required_columns = ["StockId", "Date", "raw_qit_quantum_zeno_freezing"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_quantum_zeno_freezing"]].copy()
        df.rename(columns={"raw_qit_quantum_zeno_freezing": self.name}, inplace=True)
        return df
