import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitSubadditivityEntropyGap(BaseFeature):
    name = "f_qit_subadditivity_entropy_gap"
    description = "熵次加成性裂口，衡量Tick價格跳動與成交規模分級的互資訊。"
    
    required_columns = ["StockId", "Date", "raw_qit_subadditivity_entropy_gap"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_subadditivity_entropy_gap"]].copy()
        df.rename(columns={"raw_qit_subadditivity_entropy_gap": self.name}, inplace=True)
        return df
