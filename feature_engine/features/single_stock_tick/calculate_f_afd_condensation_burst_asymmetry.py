import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdCondensationBurstAsymmetry(BaseFeature):
    name = "f_afd_condensation_burst_asymmetry"
    description = "AFD特徵：降水凝結引發的位渦相變奇點"
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
        
        df["_sell_pr_lag"] = df.groupby("StockId")["SellPr"].shift(1).fillna(df["SellPr"])
        df["_C_t"] = np.maximum(0, df["DealPrice"] - df["_sell_pr_lag"])
        df["_burst"] = df["_C_t"] * df["DealCount"]
        df["_burst_cubed"] = df["_burst"] ** 3
        
        daily_raw = df.groupby(["StockId", "Date_int"]).agg(
            raw=("_burst_cubed", "sum")
        ).reset_index()
        
        daily_raw["raw"] = np.log1p(np.abs(daily_raw["raw"])) * np.sign(daily_raw["raw"])
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
