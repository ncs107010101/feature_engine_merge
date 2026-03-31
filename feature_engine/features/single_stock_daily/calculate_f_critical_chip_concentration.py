import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureCriticalChipConcentration(BaseFeature):
    name = "f_chip_concentration"
    description = "籌碼臨界集中度 (Critical Chip Concentration)"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        chip_col = "超過1000張集保占比_最近1次發布資料"
        if chip_col in df.columns and not df[chip_col].isna().all():
            chip_pct = df[chip_col].fillna(0) / 100
        else:
            chip_pct = (df.get("外資持股率", 0).fillna(0) + 
                        df.get("投信持股率", 0).fillna(0) + 
                        df.get("自營商持股率", 0).fillna(0)) / 100
                        
        hhi = chip_pct ** 2
        
        hhi_zscore = hhi.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=60, min_periods=1))
        enhancement = np.where(hhi > 0.15, np.exp(hhi * 5), 1.0)
        
        result = hhi_zscore * enhancement
        result = result.clip(-10, 10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
