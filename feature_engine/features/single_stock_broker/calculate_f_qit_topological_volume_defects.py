import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitTopologicalVolumeDefects(BaseFeature):
    name = "f_qit_topological_volume_defects"
    description = "拓樸量能缺陷密度。買超券商均價小於VWAP的成交量佔比。"
    
    required_columns = ["StockId", "Date", "raw_qit_topological_volume_defects"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_topological_volume_defects"]].copy()
        df.rename(columns={"raw_qit_topological_volume_defects": self.name}, inplace=True)
        return df
