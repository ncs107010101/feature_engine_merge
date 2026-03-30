"""
Feature: f_be_frustration_driven_exit
Alpha v17: Frustration Driven Exit - High expectation then frustration
Theory: After high positive expectation bins, frustration exits with small sells
Direction: Positive → High frustration → Extreme LOW return
Data: single_stock_tick
Transform: ewm_then_zscore(5, 20)
"""

import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore

EPS = 1e-8


@register_feature
class FeatureBeFrustrationDrivenExit(BaseFeature):
    name = "f_be_frustration_driven_exit"
    description = "Alpha v17: High expectation (4+ positive bins) followed by small sell volume. Indicates retail frustration after failed breakout."
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
        
        thresh = data[["StockId", "Date", "Small_Thresh"]].drop_duplicates()
        df = df.merge(thresh, left_on=["StockId", "Date_int"], right_on=["StockId", "Date"], how="left")
        df["Small_Thresh"] = df["Small_Thresh"].fillna(df.groupby("Date_int")["DealCount"].transform(lambda x: x.quantile(0.3)))
        df["Small_Thresh"] = df["Small_Thresh"].fillna(1)
        
        BIN_SIZE = 200
        df["_bin"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        day_sizes = df.groupby("Date_int").transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby("Date_int").cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_small_sell"] = np.where(
            (df["PrFlag"] == 0) & (df["DealCount"] <= df["Small_Thresh"]),
            df["DealCount"], 0
        )
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            bin_open=("DealPrice", "first"),
            bin_close=("DealPrice", "last"),
            small_sell=("_small_sell", "sum"),
        ).reset_index()
        
        bin_agg["bin_ret"] = np.where(
            bin_agg["bin_open"] > 0,
            (bin_agg["bin_close"] - bin_agg["bin_open"]) / bin_agg["bin_open"],
            0
        )
        bin_agg = bin_agg.sort_values(["StockId", "Date_int", "_bin"]).reset_index(drop=True)
        
        bin_agg["pos_ret"] = (bin_agg["bin_ret"] > 0).astype(int)
        bin_agg["high_expect"] = bin_agg.groupby(["StockId", "Date_int"])["pos_ret"].transform(
            lambda x: x.rolling(5, min_periods=5).sum().shift(1)
        )
        bin_agg["high_expect"] = (bin_agg["high_expect"] >= 4).astype(int)
        bin_agg["frustration"] = (bin_agg["bin_ret"] < -0.005).astype(int)
        bin_agg["raw"] = bin_agg["high_expect"] * bin_agg["frustration"] * bin_agg["small_sell"]
        
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