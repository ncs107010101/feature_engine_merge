import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdContinuousSpectrumTransient(BaseFeature):
    name = "f_afd_continuous_spectrum_transient"
    description = "AFD特徵：擾動投射到連續譜的瞬間非模態增長"
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
        
        df["_s_cont"] = df.groupby("StockId")["DealPrice"].diff().fillna(0)
        df["_s_disc"] = df.groupby("StockId")["DealPrice"].shift(60).fillna(df["DealPrice"])
        df["_s_disc"] = df["DealPrice"] - df["_s_disc"]
        
        df["_alignment"] = np.maximum(0, df["_s_cont"] * df["_s_disc"])
        df["_weighted"] = df["_alignment"] * df["DealCount"]
        
        daily_raw = df.groupby(["StockId", "Date_int"]).agg(
            raw=("_weighted", "sum")
        ).reset_index()
        
        daily_raw["raw"] = np.log1p(daily_raw["raw"])
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
