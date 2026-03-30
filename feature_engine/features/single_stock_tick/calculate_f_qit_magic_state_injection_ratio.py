import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitMagicStateInjectionRatio(BaseFeature):
    name = "f_qit_magic_state_injection_ratio"
    description = "魔術態注入比例。一筆市價單瞬間掃透多檔委託簿的異常大單成交佔比。"
    
    required_columns = ["StockId", "Date", "raw_qit_magic_state_injection_ratio"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_magic_state_injection_ratio"]].copy()
        df.rename(columns={"raw_qit_magic_state_injection_ratio": self.name}, inplace=True)
        return df
