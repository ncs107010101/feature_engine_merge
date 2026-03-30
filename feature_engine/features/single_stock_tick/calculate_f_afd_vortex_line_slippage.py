import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdVortexLineSlippage(BaseFeature):
    name = "f_afd_vortex_line_slippage"
    description = "AFD特徵：渦線滑移"
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
        
        max_idx = df.groupby("Date_int").cumcount().groupby(df["Date_int"]).transform("max")
        cum_idx = df.groupby("Date_int").cumcount()
        
        early_mask = cum_idx < max_idx * 0.33
        mid_mask = (cum_idx >= max_idx * 0.33) & (cum_idx < max_idx * 0.66)
        late_mask = cum_idx >= max_idx * 0.66
        
        df["_dir_early"] = np.where(early_mask, df["_direction"], 0.0)
        df["_dir_mid"] = np.where(mid_mask, df["_direction"], 0.0)
        df["_dir_late"] = np.where(late_mask, df["_direction"], 0.0)
        
        def compute_slippage(group):
            early = group["_dir_early"].sum()
            mid = group["_dir_mid"].sum()
            late = group["_dir_late"].sum()
            return (late - mid) * early
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_slippage).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
