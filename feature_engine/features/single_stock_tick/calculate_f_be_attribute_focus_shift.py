import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore

EPS = 1e-8


@register_feature
class FeatureBeAttributeFocusShift(BaseFeature):
    name = "f_be_attribute_focus_shift"
    description = "Alpha v17: Negative correlation between spread and buy volume. Smart buying at low spread."
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
        
        if "BuyPr" in df.columns and "SellPr" in df.columns:
            df["_spread"] = (df["SellPr"] - df["BuyPr"]).clip(lower=0)
        else:
            df["_spread"] = 0
        
        BIN_SIZE = 100
        df["_bin"] = df.groupby(["StockId", "Date_int"]).cumcount() // BIN_SIZE
        day_sizes = df.groupby(["StockId", "Date_int"]).transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby(["StockId", "Date_int"]).cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_buy_vol"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            spread_avg=("_spread", "mean"),
            buy_vol=("_buy_vol", "sum"),
        ).reset_index()
        
        def _corr(group):
            if len(group) < 5:
                return np.nan
            return group["spread_avg"].corr(group["buy_vol"])
        
        daily_corr = bin_agg.groupby(["StockId", "Date_int"]).apply(_corr).reset_index()
        daily_corr.columns = ["StockId", "Date_int", "corr_val"]
        daily_corr = daily_corr.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        daily_corr["raw"] = -daily_corr["corr_val"]
        
        out_series = daily_corr.groupby("StockId")["raw"].transform(
            lambda x: ewm_then_zscore(x, ewm_span=5, z_window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": daily_corr["StockId"],
            "Date": daily_corr["Date_int"],
            self.name: out.values
        })