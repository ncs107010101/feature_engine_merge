import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitArnoldWebResonance(BaseFeature):
    name = "f_qit_arnold_web_resonance"
    description = "阿諾德網共振指數。單一價位最高成交量(Mode Price)佔全日總量的比例。"
    
    required_columns = ["StockId", "Date", "raw_qit_arnold_web_resonance"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_arnold_web_resonance"]].copy()
        df.rename(columns={"raw_qit_arnold_web_resonance": self.name}, inplace=True)
        return df
