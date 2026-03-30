import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

EPS = 1e-8


@register_feature
class FeatureGtCooperativeDefectionThreshold(BaseFeature):
    name = "f_gt_cooperative_defection_threshold"
    description = "Alpha v17: Max sell/buy ratio when cooperation above mean and sell > 1.5x cooperation. Cooperative defection."
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
        
        BIN_SIZE = 100
        df["_bin"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        day_sizes = df.groupby("Date_int").transform("size")
        n_full = (day_sizes // BIN_SIZE) * BIN_SIZE
        valid_mask = df.groupby("Date_int").cumcount() < n_full
        df.loc[~valid_mask, "_bin"] = -1
        
        df["_buy_vol"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
        df["_sell_vol"] = np.where(df["PrFlag"] == 0, df["DealCount"], 0)
        
        bin_agg = df[df["_bin"] >= 0].groupby(["StockId", "Date_int", "_bin"]).agg(
            buy_vol=("_buy_vol", "sum"),
            sell_vol=("_sell_vol", "sum"),
        ).reset_index()
        bin_agg = bin_agg.sort_values(["StockId", "Date_int", "_bin"]).reset_index(drop=True)
        
        records = []
        for (stock_id, date), grp in bin_agg.groupby(["StockId", "Date_int"]):
            if len(grp) < 21:
                records.append({"StockId": stock_id, "Date_int": date, "raw": 0.0})
                continue
            grp = grp.reset_index(drop=True)
            coop = grp["buy_vol"].rolling(20, min_periods=5).sum()
            day_mean = coop.mean()
            max_raw = 0.0
            for i in range(20, len(grp)):
                c = coop.iloc[i - 1] if i > 0 else 0
                s = grp.iloc[i]["sell_vol"]
                if c > day_mean and s > c * 1.5:
                    val = s / (c + EPS)
                    max_raw = max(max_raw, val)
            records.append({"StockId": stock_id, "Date_int": date, "raw": max_raw})
        
        df_result = pd.DataFrame(records)
        df_result = df_result.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = df_result.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, 20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df_result["StockId"],
            "Date": df_result["Date_int"],
            self.name: out.values
        })