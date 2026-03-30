"""
Feature: f_be_decisive_execution_premium
Alpha v17: Decisive Execution Premium - Early bin decisive execution ratio
Theory: Early buy volume ratio indicates institutional conviction
Direction: Positive → High early execution → Extreme HIGH return
Data: single_stock_tick
Transform: ewm_then_zscore(5, 20)
"""

import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore

EPS = 1e-8


@register_feature
class FeatureBeDecisiveExecutionPremium(BaseFeature):
    name = "f_be_decisive_execution_premium"
    description = "Alpha v17: Early bin (first 10 ticks) decisive execution ratio. High early buy volume indicates strong institutional conviction."
    required_columns = ["StockId", "Date", "Small_Thresh", "Large_Thresh"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            tick_raw = data
        
        df = tick_raw.copy()
        
        date_col = "Date" if "Date" in df.columns else "Date_int"
        df["Date"] = df[date_col].astype(int)
        df = df.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
        
        df["Date_int"] = df["Date"]
        
        thresh = data[["StockId", "Date", "Small_Thresh", "Large_Thresh"]].drop_duplicates()
        df = df.merge(thresh, left_on=["StockId", "Date_int"], right_on=["StockId", "Date"], how="left")
        df["Large_Thresh"] = df["Large_Thresh"].fillna(df.groupby("Date_int")["DealCount"].transform(lambda x: x.quantile(0.8)))
        df["Large_Thresh"] = df["Large_Thresh"].fillna(1)
        
        BIN_SIZE = 100
        df["_bin"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        day_sizes = df.groupby("Date_int").transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby("Date_int").cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_intra_idx"] = df.groupby(["Date_int", "_bin"]).cumcount()
        df["_buy_vol"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
        
        early_agg = df[(df["_bin"] >= 0) & (df["_intra_idx"] < 10)].groupby(["StockId", "Date_int"]).agg(
            early_buy=("_buy_vol", "sum")
        ).reset_index()
        
        total_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int"]).agg(
            total_buy=("_buy_vol", "sum")
        ).reset_index()
        
        bins = early_agg.merge(total_agg, on=["StockId", "Date_int"])
        bins["decisive_ratio"] = bins["early_buy"] / (bins["total_buy"] + EPS)
        
        daily_raw = bins.groupby(["StockId", "Date_int"])["decisive_ratio"].max().reset_index()
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["decisive_ratio"].transform(
            lambda x: ewm_then_zscore(x, ewm_span=5, z_window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": daily_raw["StockId"],
            "Date": daily_raw["Date_int"],
            self.name: out.values
        })