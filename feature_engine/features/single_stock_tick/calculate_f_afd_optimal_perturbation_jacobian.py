import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdOptimalPerturbationJacobian(BaseFeature):
    name = "f_afd_optimal_perturbation_jacobian"
    description = "AFD特徵：最優擾動Jacobian"
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
        df["_price_diff"] = df.groupby("StockId")["DealPrice"].diff().fillna(0)
        
        df["_bin_id"] = df.groupby("Date_int").cumcount() // 50
        
        def compute_jacobian(group):
            direction = group["_direction"].values
            price_diff = group["_price_diff"].values
            
            n_bins = len(direction) // 50
            if n_bins < 2:
                return 0.0
            
            A = np.zeros(n_bins)
            B = np.zeros(n_bins)
            
            for i in range(n_bins):
                s, e = i * 50, (i + 1) * 50
                A[i] = np.sum(direction[s:e])
                B[i] = np.sum(price_diff[s:e])
            
            det = A[:-1] * np.diff(B) - B[:-1] * np.diff(A)
            return np.sum(det)
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_jacobian).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
