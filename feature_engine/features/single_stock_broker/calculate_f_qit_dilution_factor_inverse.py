import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitDilutionFactorInverse(BaseFeature):
    name = "f_qit_dilution_factor_inverse"
    description = "參與態稀釋因子倒數。各券商成交量佔比平方和 (HHI)。"
    
    required_columns = ["StockId", "Date", "raw_qit_dilution_factor_inverse"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_dilution_factor_inverse"]].copy()
        df.rename(columns={"raw_qit_dilution_factor_inverse": self.name}, inplace=True)
        return df
