import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitLyapunovBallisticDuration(BaseFeature):
    name = "f_qit_lyapunov_ballistic_duration"
    description = "彈道傳播期持續時間。最長的純主動買進連續Tick序列的累積成交量。"
    
    required_columns = ["StockId", "Date", "raw_qit_lyapunov_ballistic_duration"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_lyapunov_ballistic_duration"]].copy()
        df.rename(columns={"raw_qit_lyapunov_ballistic_duration": self.name}, inplace=True)
        return df
