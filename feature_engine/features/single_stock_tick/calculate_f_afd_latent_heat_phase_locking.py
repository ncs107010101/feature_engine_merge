import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

BIN_SIZE = 30


@register_feature
class FeatureAfdLatentHeatPhaseLocking(BaseFeature):
    name = "f_afd_latent_heat_phase_locking"
    description = "AFD特徵：DRV低層正渦度與中層負渦度的相位鎖定"
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
        
        df["_buy_seq"] = np.where(df["PrFlag"] == 1, df["DealCount"].astype(float), 0.0)
        df["_sell_seq"] = np.where(df["PrFlag"] == 0, df["DealCount"].astype(float), 0.0)
        
        df["_bin_num"] = df.groupby("Date_int").cumcount()
        df["_bin_id"] = df["_bin_num"] // BIN_SIZE
        df["_is_full_bin"] = df["_bin_num"] < (df.groupby("Date_int")["_bin_num"].transform("max") // BIN_SIZE * BIN_SIZE)
        
        df_full = df[df["_is_full_bin"]].copy()
        
        def compute_inner_product(group):
            buy = group["_buy_seq"].values
            sell = group["_sell_seq"].values
            price = group["DealPrice"].values
            if len(buy) < 5:
                return np.nan
            inner = np.dot(buy, sell)
            x = np.arange(len(price), dtype=np.float64)
            if np.std(x) > 0 and np.std(price) > 0:
                slope = np.polyfit(x, price, 1)[0]
            else:
                slope = 0.0
            return inner * max(0.0, np.sign(slope))
        
        if len(df_full) > 0:
            bin_agg = df_full.groupby(["StockId", "Date_int", "_bin_id"]).apply(compute_inner_product).reset_index(name="val")
            bin_agg = bin_agg.dropna(subset=["val"])
            daily_raw = bin_agg.groupby(["StockId", "Date_int"])["val"].sum().reset_index()
        else:
            daily_raw = pd.DataFrame(columns=["StockId", "Date_int", "val"])
        
        all_dates = df.groupby(["StockId", "Date_int"]).size().reset_index()[["StockId", "Date_int"]]
        daily_raw = all_dates.merge(daily_raw, on=["StockId", "Date_int"], how="left")
        daily_raw["val"] = daily_raw["val"].fillna(0)
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["val"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
