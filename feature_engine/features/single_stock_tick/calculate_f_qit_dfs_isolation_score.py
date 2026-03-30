import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitDfsIsolationScore(BaseFeature):
    name = "f_qit_dfs_isolation_score"
    description = "去相干無噪子空間隔離度。點差內交易(不主動外打)的成交量佔比。"
    
    required_columns = ["StockId", "Date", "raw_qit_dfs_isolation_score"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_dfs_isolation_score"]].copy()
        df.rename(columns={"raw_qit_dfs_isolation_score": self.name}, inplace=True)
        return df
