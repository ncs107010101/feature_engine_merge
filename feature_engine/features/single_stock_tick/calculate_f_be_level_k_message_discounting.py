import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore

EPS = 1e-8


@register_feature
class FeatureBeLevelKMessageDiscounting(BaseFeature):
    name = "f_be_level_k_message_discounting"
    description = "Alpha v17: Top 10% buy volume bins with VWAP discount. Level-k message discounting at low VWAP."
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
        
        BIN_SIZE = 100
        df["_bin"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        day_sizes = df.groupby("Date_int").transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby("Date_int").cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_buy_vol"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            buy_vol=("_buy_vol", "sum"),
            vwap_num=("_pv", "sum"),
            vwap_den=("DealCount", "sum"),
            bin_open=("DealPrice", "first"),
        ).reset_index()
        bin_agg["vwap"] = bin_agg["vwap_num"] / (bin_agg["vwap_den"] + EPS)
        
        q90 = bin_agg.groupby("Date_int")["buy_vol"].transform(lambda x: x.quantile(0.9))
        bin_agg["l1_signal"] = (bin_agg["buy_vol"] > q90).astype(int)
        bin_agg["l2_discount"] = (
            (bin_agg["vwap"] < bin_agg["bin_open"]).astype(int) * 
            (bin_agg["vwap"] / (bin_agg["bin_open"] + EPS) - 1).abs()
        )
        bin_agg["raw"] = bin_agg["l1_signal"] * bin_agg["l2_discount"]
        
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