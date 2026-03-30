import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdMoistBaroclinicAcceleration(BaseFeature):
    name = "f_afd_moist_baroclinic_acceleration"
    description = "AFD特徵：濕對流斜壓加速度"
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
        
        df["_E_m"] = np.where(df["PrFlag"] == 1, df["DealCount"].astype(float), 0.0)
        df["_E_d"] = np.where(df["PrFlag"] == 0, df["DealCount"].astype(float), 0.0)
        
        split_idx = (df.groupby("Date_int").cumcount() * 2) >= df.groupby("Date_int").cumcount().groupby(df["Date_int"]).transform("max")
        
        df["_E_m_early"] = np.where(split_idx, df["_E_m"], 0.0)
        df["_E_m_late"] = np.where(~split_idx, df["_E_m"], 0.0)
        df["_E_d_early"] = np.where(split_idx, df["_E_d"], 0.0)
        df["_E_d_late"] = np.where(~split_idx, df["_E_d"], 0.0)
        
        def compute_baroclinic(group):
            E_m_e = group["_E_m_early"].sum()
            E_d_e = group["_E_d_early"].sum()
            E_m_l = group["_E_m_late"].sum()
            E_d_l = group["_E_d_late"].sum()
            
            if E_m_e + E_d_e == 0:
                return 0.0
            return (E_m_l - E_d_l) / (E_m_e + E_d_e + 1e-10)
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_baroclinic).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
