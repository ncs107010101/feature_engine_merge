import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FIntraguildPredationSkew(BaseFeature):
    name = "f_intraguild_predation_skew"
    description = "競爭排除與同類相食 - 最強Top5大戶淨買入/第6~15名券商淨賣出的比例"
    required_columns = ["StockId", "Date", "raw_top5_net_buy", "raw_mid10_net_sell"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        raw_val = df["raw_top5_net_buy"] / (df["raw_mid10_net_sell"] + eps)
        
        out_series = g["raw_top5_net_buy"].transform(lambda x: zscore_rolling(raw_val.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
