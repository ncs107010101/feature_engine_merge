import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

EPS = 1e-8


@register_feature
class FeatureBeEgoInformationAvoidance(BaseFeature):
    name = "f_be_ego_information_avoidance"
    description = "Alpha v17: Volume avoidance when price is near floor. Retail avoids low-priced zones."
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
        
        if "FloorPr" not in df.columns:
            df["FloorPr"] = df.groupby(["StockId", "Date_int"])["DealPrice"].transform("min")
        
        BIN_SIZE = 50
        df["_bin"] = df.groupby(["StockId", "Date_int"]).cumcount() // BIN_SIZE
        day_sizes = df.groupby(["StockId", "Date_int"]).transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby(["StockId", "Date_int"]).cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_close_to_floor"] = (df["DealPrice"] <= df["FloorPr"] * 1.02).astype(int)
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            bin_vol=("DealCount", "sum"),
            has_floor=("_close_to_floor", "max"),
        ).reset_index()
        
        avg_bin_vol = bin_agg.groupby(["StockId", "Date_int"])["bin_vol"].transform("mean")
        bin_agg["avoidance"] = (1 - bin_agg["bin_vol"] / (avg_bin_vol + EPS)) * bin_agg["has_floor"]
        
        daily_raw = bin_agg.groupby(["StockId", "Date_int"])["avoidance"].max().reset_index()
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["avoidance"].transform(
            lambda x: zscore_rolling(x, 20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": daily_raw["StockId"],
            "Date": daily_raw["Date_int"],
            self.name: out.values
        })