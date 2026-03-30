import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FChipLargeHolderPct(BaseFeature):
    name = "f_chip_large_holder_pct"
    description = "Large holder percentage"
    required_columns = ["StockId", "Date", "超過1000張集保占比_最近1次發布資料"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        # Post-processing (ffill by StockId for weekly-updated data)
        out = df["超過1000張集保占比_最近1次發布資料"].copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
