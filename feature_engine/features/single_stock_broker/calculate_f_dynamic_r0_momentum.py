import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FDynamicR0Momentum(BaseFeature):
    name = "f_dynamic_r0_momentum"
    description = "動態R0動能 - 當日大戶淨買超/總成交量 - 前5日均值"
    required_columns = ["StockId", "Date", "raw_top10_net_buy", "_total_net"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        eps = 1e-8
        r0 = df["raw_top10_net_buy"] / (df["_total_net"] + eps)
        
        df_r0 = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            "r0": r0
        }).set_index(["StockId", "Date"]).sort_index()
        
        r0_ma5 = df_r0["r0"].groupby(level="StockId").shift(1).rolling(5, min_periods=1).mean()
        raw_val = df_r0["r0"] - r0_ma5
        raw_val_ewm = raw_val.ewm(span=5, min_periods=1).mean()
        
        df["raw_val_ewm"] = raw_val_ewm.values
        
        out_series = g["raw_top10_net_buy"].transform(lambda x: zscore_rolling(df["raw_val_ewm"].loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
