import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore

EPS = 1e-8


@register_feature
class FeatureBeRedundancyNeglectCascade(BaseFeature):
    name = "f_be_redundancy_neglect_cascade"
    description = "Alpha v17: Up cascade (3+ consecutive positive bins) with small/large buy ratio. Redundancy neglect in momentum."
    required_columns = ["StockId", "Date", "Small_Thresh", "Large_Thresh"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            tick_raw = data
        
        df = tick_raw.copy()
        df["Date"] = df["Date"].astype(int)
        df = df.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
        
        df["Date_int"] = df["Date"]
        
        thresh = data[["StockId", "Date", "Small_Thresh", "Large_Thresh"]].drop_duplicates()
        df = df.merge(thresh, left_on=["StockId", "Date_int"], right_on=["StockId", "Date"], how="left")
        df["Small_Thresh"] = df["Small_Thresh"].fillna(1)
        df["Large_Thresh"] = df["Large_Thresh"].fillna(df.groupby("Date_int")["DealCount"].transform(lambda x: x.quantile(0.8)))
        df["Large_Thresh"] = df["Large_Thresh"].fillna(1)
        
        BIN_SIZE = 50
        df["_bin"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        day_sizes = df.groupby("Date_int").transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby("Date_int").cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_small_buy"] = np.where(
            (df["PrFlag"] == 1) & (df["DealCount"] <= df["Small_Thresh"]),
            df["DealCount"], 0
        )
        df["_large_buy"] = np.where(
            (df["PrFlag"] == 1) & (df["DealCount"] >= df["Large_Thresh"]),
            df["DealCount"], 0
        )
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            bin_open=("DealPrice", "first"),
            bin_close=("DealPrice", "last"),
            small_buy=("_small_buy", "sum"),
            large_buy=("_large_buy", "sum"),
        ).reset_index()
        
        bin_agg["bin_ret"] = np.where(
            bin_agg["bin_open"] > 0,
            (bin_agg["bin_close"] - bin_agg["bin_open"]) / bin_agg["bin_open"],
            0
        )
        bin_agg = bin_agg.sort_values(["StockId", "Date_int", "_bin"]).reset_index(drop=True)
        
        bin_agg["pos"] = (bin_agg["bin_ret"] > 0).astype(int)
        bin_agg["up_cascade"] = bin_agg.groupby("Date_int")["pos"].transform(
            lambda x: x.rolling(3, min_periods=3).sum()
        )
        bin_agg["up_cascade"] = (bin_agg["up_cascade"] == 3).astype(int)
        bin_agg["redundancy"] = bin_agg["small_buy"] / (bin_agg["large_buy"] + EPS)
        bin_agg["raw"] = bin_agg["up_cascade"] * bin_agg["redundancy"]
        
        daily_raw = bin_agg.groupby(["StockId", "Date_int"])["raw"].max().reset_index()
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: ewm_then_zscore(x, ewm_span=5, z_window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": daily_raw["StockId"],
            "Date": daily_raw["Date_int"],
            self.name: out.values
        })