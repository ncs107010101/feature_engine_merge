import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitTeleportationFidelity(BaseFeature):
    name = "f_qit_teleportation_fidelity"
    description = "狀態傳態保真度。跨時空的狀態複製，計算早尾盤1分鐘K棒報酬率的餘弦相似度。"
    
    required_columns = ["StockId", "Date", "raw_qit_teleportation_fidelity"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_teleportation_fidelity"]].copy()
        df.rename(columns={"raw_qit_teleportation_fidelity": self.name}, inplace=True)
        return df
