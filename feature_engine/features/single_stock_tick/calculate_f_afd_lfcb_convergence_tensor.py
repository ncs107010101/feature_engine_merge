import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdLfcbConvergenceTensor(BaseFeature):
    name = "f_afd_lfcb_convergence_tensor"
    description = "AFD特徵：LFCB收斂張量"
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
        
        df["_direction"] = np.where(df["PrFlag"] == 1, 1.0, np.where(df["PrFlag"] == 0, -1.0, 0.0))
        
        df["_bin_id"] = df.groupby("Date_int").cumcount() // 100
        
        def compute_convergence(group):
            direction = group["_direction"].values
            
            n_bins = len(direction) // 100
            if n_bins < 2:
                return 0.0
            
            conv = 0.0
            for i in range(n_bins - 1):
                s1, e1 = i * 100, (i + 1) * 100
                s2, e2 = (i + 1) * 100, (i + 2) * 100
                
                flow1 = np.sum(direction[s1:e1])
                flow2 = np.sum(direction[s2:e2])
                
                conv += flow1 * flow2
            
            return conv
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_convergence).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
