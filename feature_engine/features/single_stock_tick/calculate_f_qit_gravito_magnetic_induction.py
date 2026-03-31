import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitGravitoMagneticInduction(BaseFeature):
    name = "f_qit_gravito_magnetic_induction"
    description = "重力磁感應動量。分鐘內的主動買進速率乘以收盤價與VWAP的距離。"
    
    required_columns = ["StockId", "Date", "raw_qit_gravito_magnetic_induction"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_gravito_magnetic_induction"]].copy()
        df.rename(columns={"raw_qit_gravito_magnetic_induction": self.name}, inplace=True)
        return df
