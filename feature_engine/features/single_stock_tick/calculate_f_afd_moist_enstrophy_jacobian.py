import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

BIN_SIZE = 50


@register_feature
class FeatureAfdMoistEnstrophyJacobian(BaseFeature):
    name = "f_afd_moist_enstrophy_jacobian"
    description = "AFD特徵：濕對流渦度擬能Jacobian"
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
        
        df["_bin_id"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        
        def compute_jacobian(group):
            direction = group["_direction"].values
            price_diff = group["_price_diff"].values
            
            n_bins = len(direction) // BIN_SIZE
            if n_bins < 2:
                return 0.0
            
            A = np.zeros(n_bins)
            B = np.zeros(n_bins)
            
            for i in range(n_bins):
                s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
                A[i] = np.sum(direction[s:e])
                B[i] = np.sum(price_diff[s:e])
            
            if len(A) < 2 or len(B) < 2:
                return 0.0
            
            J = A[:-1] * np.diff(B) - B[:-1] * np.diff(A)
            return np.sum(J)
        
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
