import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

EPS = 1e-8


@register_feature
class FeatureBeUnobservableDownCascade(BaseFeature):
    name = "f_be_unobservable_down_cascade"
    description = "Alpha v17: Drought (low buy volume) combined with bleeding (close < open). Unobservable cascade pattern."
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
        
        BIN_SIZE = 200
        df["_bin"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        day_sizes = df.groupby("Date_int").transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby("Date_int").cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_buy_vol"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            active_buy=("_buy_vol", "sum"),
            bin_open=("DealPrice", "first"),
            bin_close=("DealPrice", "last"),
        ).reset_index()
        
        q10 = bin_agg.groupby("Date_int")["active_buy"].transform(lambda x: x.quantile(0.1))
        bin_agg["drought"] = (bin_agg["active_buy"] < q10).astype(int)
        bin_agg["bleeding"] = (bin_agg["bin_close"] < bin_agg["bin_open"]).astype(int)
        bin_agg["cascade"] = bin_agg["drought"] * bin_agg["bleeding"]
        
        daily_raw = bin_agg.groupby(["StockId", "Date_int"])["cascade"].sum().reset_index()
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["cascade"].transform(
            lambda x: zscore_rolling(x, 20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": daily_raw["StockId"],
            "Date": daily_raw["Date_int"],
            self.name: out.values
        })