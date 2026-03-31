import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

EPS = 1e-8


@register_feature
class FeatureGtOptimalDeliberationRisk(BaseFeature):
    name = "f_gt_optimal_deliberation_risk"
    description = "Alpha v17: Optimal deliberation risk - delay ratio (argmax bin / max bin). Late peak deliberation."
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
        
        BIN_SIZE = 100
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
            large_buy=("_large_buy", "sum"),
        ).reset_index()
        
        idx_max = bin_agg.groupby("Date_int")["large_buy"].idxmax()
        best_bins = bin_agg.loc[idx_max][["Date_int", "_bin"]].copy()
        best_bins.columns = ["Date_int", "argmax_bin"]
        
        total_bins = bin_agg.groupby("Date_int")["_bin"].max().reset_index()
        total_bins.columns = ["Date_int", "max_bin"]
        
        df_result = best_bins.merge(total_bins, on="Date_int")
        df_result["delay_ratio"] = df_result["argmax_bin"] / (df_result["max_bin"] + EPS)
        df_result["raw"] = 1 - df_result["delay_ratio"]
        df_result = df_result.sort_values("Date_int").reset_index(drop=True)
        
        out_series = df_result.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, 20)
        ) if "StockId" in df_result.columns else df_result["raw"].transform(
            lambda x: zscore_rolling(x, 20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": "Unknown",
            "Date": df_result["Date_int"],
            self.name: out.values
        })