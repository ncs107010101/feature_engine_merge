import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

EPS = 1e-8


@register_feature
class FeatureGtExtremistSeparatingHigh(BaseFeature):
    name = "f_gt_extremist_separating_high"
    description = "Alpha v17: Large buys at daily high price zones. Extremist separating behavior at peaks."
    required_columns = ["StockId", "Date", "Large_Thresh"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            tick_raw = data
        
        df = tick_raw.copy()
        df["Date"] = df["Date"].astype(int)
        df = df.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
        
        df["Date_int"] = df["Date"]
        
        thresh = data[["StockId", "Date", "Large_Thresh"]].drop_duplicates()
        df = df.merge(thresh, left_on=["StockId", "Date_int"], right_on=["StockId", "Date"], how="left")
        df["Large_Thresh"] = df["Large_Thresh"].fillna(df.groupby("Date_int")["DealCount"].transform(lambda x: x.quantile(0.8)))
        df["Large_Thresh"] = df["Large_Thresh"].fillna(1)
        
        BIN_SIZE = 50
        df["_bin"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        day_sizes = df.groupby("Date_int").transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby("Date_int").cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_large_buy"] = np.where(
            (df["PrFlag"] == 1) & (df["DealCount"] >= df["Large_Thresh"]),
            df["DealCount"], 0
        )
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            bin_high=("DealPrice", "max"),
            large_buy=("_large_buy", "sum"),
        ).reset_index()
        
        bin_agg = bin_agg.sort_values(["StockId", "Date_int", "_bin"]).reset_index(drop=True)
        bin_agg["daily_high"] = bin_agg.groupby("Date_int")["bin_high"].cummax()
        bin_agg["is_extremist"] = (bin_agg["bin_high"] >= bin_agg["daily_high"]).astype(int)
        bin_agg["signal"] = bin_agg["is_extremist"] * bin_agg["large_buy"]
        
        daily_raw = bin_agg.groupby(["StockId", "Date_int"])["signal"].sum().reset_index()
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["signal"].transform(
            lambda x: zscore_rolling(x, 20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": daily_raw["StockId"],
            "Date": daily_raw["Date_int"],
            self.name: out.values
        })