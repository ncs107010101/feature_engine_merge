import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdBifurcationTippingPoint(BaseFeature):
    name = "f_afd_bifurcation_tipping_point"
    description = "AFD特徵：臨界分岔點"
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
        
        df["_bin_id"] = df.groupby("Date_int").cumcount() // BIN_SIZE
        
        def compute_bifurcation(group):
            n = len(group)
            if n < BIN_SIZE:
                return 0.0
            
            dc = group["DealCount"].values.astype(np.float64)
            pf = group["PrFlag"].values
            dp = group["DealPrice"].values
            
            n_bins = n // BIN_SIZE
            if n_bins < 2:
                return 0.0
            
            energies = np.zeros(n_bins)
            dp_bins = np.zeros(n_bins)
            for i in range(n_bins):
                s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
                buy_f = np.sum(dc[s:e][pf[s:e] == 1]) / BIN_SIZE
                sell_f = np.sum(dc[s:e][pf[s:e] == 0]) / BIN_SIZE
                p_s = dp[s:e]
                x = np.arange(len(p_s), dtype=np.float64)
                if len(p_s) > 5:
                    sl5 = np.polyfit(x[:5], p_s[:5], 1)[0]
                    sl30 = np.polyfit(x[:min(30, len(x))], p_s[:min(30, len(p_s))], 1)[0]
                else:
                    sl5, sl30 = 0.0, 0.0
                
                energies[i] = buy_f ** 2 + sell_f ** 2 + sl5 ** 2 + sl30 ** 2
                dp_bins[i] = dp[e - 1] - dp[s]
            
            threshold = np.mean(energies) + 2 * np.std(energies)
            exceed = np.maximum(0, energies - threshold)
            dir_sign = np.sign(np.maximum(0, dp_bins))
            return np.sum(exceed * dir_sign)
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_bifurcation).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=42)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
