import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitHawkingEvaporationRate(BaseFeature):
    name = "f_qit_hawking_evaporation_rate"
    description = "極值邊界霍金蒸發率。全日觸碰最高價區間時，伴隨主動賣單倒貨的頻率。"
    
    required_columns = ["StockId", "Date", "raw_qit_hawking_evaporation_rate"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_hawking_evaporation_rate"]].copy()
        df.rename(columns={"raw_qit_hawking_evaporation_rate": self.name}, inplace=True)
        return df
