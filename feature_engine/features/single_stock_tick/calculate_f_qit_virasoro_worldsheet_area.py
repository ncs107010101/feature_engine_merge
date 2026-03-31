import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitVirasoroWorldsheetArea(BaseFeature):
    name = "f_qit_virasoro_worldsheet_area"
    description = "弦論世界面幾何面積。價格路徑所包圍的面積佔理想矩形面積的比例。"
    
    required_columns = ["StockId", "Date", "raw_qit_virasoro_worldsheet_area"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_virasoro_worldsheet_area"]].copy()
        df.rename(columns={"raw_qit_virasoro_worldsheet_area": self.name}, inplace=True)
        return df
