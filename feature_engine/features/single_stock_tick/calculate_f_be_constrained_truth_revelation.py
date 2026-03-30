import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore

EPS = 1e-8


@register_feature
class FeatureBeConstrainedTruthRevelation(BaseFeature):
    name = "f_be_constrained_truth_revelation"
    description = "Alpha v17: Large buy-sell net in low spread (constrained) environment. Truth signal in tight markets."
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
        
        if "BuyPr" in df.columns and "SellPr" in df.columns:
            df["_spread"] = (df["SellPr"] - df["BuyPr"]).clip(lower=0)
        else:
            df["_spread"] = 0
        
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
        df["_large_sell"] = np.where(
            (df["PrFlag"] == 0) & (df["DealCount"] >= df["Large_Thresh"]),
            df["DealCount"], 0
        )
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            spread_mean=("_spread", "mean"),
            large_buy=("_large_buy", "sum"),
            large_sell=("_large_sell", "sum"),
        ).reset_index()
        
        q10 = bin_agg.groupby("Date_int")["spread_mean"].transform(lambda x: x.quantile(0.1))
        bin_agg["constrained"] = (bin_agg["spread_mean"] <= q10 * 1.05).astype(int)
        bin_agg["true_signal"] = bin_agg["large_buy"] - bin_agg["large_sell"]
        bin_agg["raw"] = bin_agg["constrained"] * bin_agg["true_signal"]
        
        daily_raw = bin_agg.groupby(["StockId", "Date_int"])["raw"].sum().reset_index()
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