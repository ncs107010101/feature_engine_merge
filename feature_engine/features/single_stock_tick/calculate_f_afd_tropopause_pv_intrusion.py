import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdTropopausePvIntrusion(BaseFeature):
    name = "f_afd_tropopause_pv_intrusion"
    description = "AFD特徵：對流層頂位渦侵入"
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
        
        split1 = df.groupby("Date_int").cumcount() < (df.groupby("Date_int").cumcount().groupby(df["Date_int"]).transform("max") * 0.33)
        split2 = (df.groupby("Date_int").cumcount() >= (df.groupby("Date_int").cumcount().groupby(df["Date_int"]).transform("max") * 0.33)) & \
                 (df.groupby("Date_int").cumcount() < (df.groupby("Date_int").cumcount().groupby(df["Date_int"]).transform("max") * 0.66))
        
        df["_dir_early"] = np.where(split1, df["_direction"], 0.0)
        df["_dir_mid"] = np.where(split2, df["_direction"], 0.0)
        
        def compute_intrusion(group):
            early = group["_dir_early"].sum()
            mid = group["_dir_mid"].sum()
            return early * mid
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_intrusion).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
