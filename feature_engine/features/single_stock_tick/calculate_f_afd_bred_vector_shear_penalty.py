import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdBredVectorShearPenalty(BaseFeature):
    name = "f_afd_bred_vector_shear_penalty"
    description = "AFD特徵：繁殖向量的負向剪切"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            tick_raw = data
        
        df = tick_raw.copy()
        df["Date"] = df["Date"].astype(int)
        df = df.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
        df["Date_int"] = df["Date"]
        
        def compute_shear(group):
            n = len(group)
            if n < 20:
                return 0.0
            
            dc = group["DealCount"].values.astype(np.float64)
            pf = group["PrFlag"].values
            dp = group["DealPrice"].values
            
            split = int(n * 0.6)
            direction = np.where(pf == 1, 1.0, np.where(pf == 0, -1.0, 0.0))
            
            h_flow = np.sum(dc[:split] * direction[:split])
            h_price = dp[split - 1] - dp[0]
            n_flow = np.sum(dc[split:] * direction[split:])
            n_price = dp[-1] - dp[split]
            
            dot = h_flow * n_flow + h_price * n_price
            cross = abs(h_flow * n_price - h_price * n_flow)
            
            dir_sign = np.sign(dp[-1] - dp[0])
            raw_val = cross * min(0, dot) * dir_sign
            
            return np.log1p(np.abs(raw_val)) * np.sign(raw_val)
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_shear).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=42)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
