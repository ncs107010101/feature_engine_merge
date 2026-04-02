import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FAlleeReproductionDeficit(BaseFeature):
    name = "f_allee_reproduction_deficit"
    description = "族群繁衍臨界赤字 - 大戶日買入張數的20日均值回歸程度，除以散戶賣出量"
    required_columns = ["StockId", "Date", "raw_top10_buy", "raw_bot80_sell"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.sort_values(["StockId", "Date"]).reset_index(drop = True).copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        top10_buy_mean20 = g["raw_top10_buy"].transform(lambda x: x.rolling(20, min_periods=1).mean())
        raw_val = (df["raw_top10_buy"] - top10_buy_mean20) / (df["raw_bot80_sell"] + eps)
        
        out_series = g["raw_top10_buy"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
