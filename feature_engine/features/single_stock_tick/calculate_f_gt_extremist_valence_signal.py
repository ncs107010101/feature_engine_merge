import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore

EPS = 1e-8


@register_feature
class FeatureGtExtremistValenceSignal(BaseFeature):
    name = "f_gt_extremist_valence_signal"
    description = "Alpha v17: Large buys below extreme low VWAP vs above. Sentiment valence at extremes."
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
        
        df["_pv"] = df["DealPrice"] * df["DealCount"]
        
        BIN_SIZE = 200
        df["_bin"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        day_sizes = df.groupby("Date_int").transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby("Date_int").cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            vwap_num=("_pv", "sum"),
            vwap_den=("DealCount", "sum"),
            price_std=("DealPrice", "std"),
        ).reset_index()
        bin_agg["vwap"] = bin_agg["vwap_num"] / (bin_agg["vwap_den"] + EPS)
        bin_agg["price_std"] = bin_agg["price_std"].fillna(0)
        bin_agg["extreme_low"] = bin_agg["vwap"] - 1.5 * bin_agg["price_std"]
        
        df = df.merge(bin_agg[["StockId", "Date_int", "_bin", "extreme_low"]], on=["StockId", "Date_int", "_bin"], how="left")
        
        large_buy_mask = (df["PrFlag"] == 1) & (df["DealCount"] >= df["Large_Thresh"])
        
        extremist = df[large_buy_mask & (df["DealPrice"] < df["extreme_low"])]
        centrist = df[large_buy_mask & (df["DealPrice"] >= df["extreme_low"])]
        
        ext_vol = extremist.groupby(["StockId", "Date_int", "_bin"])["DealCount"].sum().reset_index()
        ext_vol.columns = ["StockId", "Date_int", "_bin", "ext_buy"]
        
        cen_vol = centrist.groupby(["StockId", "Date_int", "_bin"])["DealCount"].sum().reset_index()
        cen_vol.columns = ["StockId", "Date_int", "_bin", "cen_buy"]
        
        bins = bin_agg[["StockId", "Date_int", "_bin"]].merge(ext_vol, on=["StockId", "Date_int", "_bin"], how="left")
        bins = bins.merge(cen_vol, on=["StockId", "Date_int", "_bin"], how="left").fillna(0)
        bins["valence"] = bins["ext_buy"] / (bins["cen_buy"] + EPS)
        
        daily_raw = bins.groupby(["StockId", "Date_int"])["valence"].mean().reset_index()
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["valence"].transform(
            lambda x: ewm_then_zscore(x, ewm_span=5, z_window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": daily_raw["StockId"],
            "Date": daily_raw["Date_int"],
            self.name: out.values
        })