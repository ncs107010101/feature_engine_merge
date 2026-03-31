import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FStefanBoundarySacrificeRatio(BaseFeature):
    name = "f_stefan_boundary_sacrifice_ratio"
    description = "史蒂芬邊界獻祭比例 - 大戶在創新高價位的買量佔其總買量的比例"
    required_columns = ["StockId", "Date", "raw_top10_boundary_buy", "raw_top10_total_buy"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        raw = df["raw_top10_boundary_buy"] / (df["raw_top10_total_buy"] + eps)
        
        out_series = g["raw_top10_boundary_buy"].transform(lambda x: zscore_rolling(raw.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
