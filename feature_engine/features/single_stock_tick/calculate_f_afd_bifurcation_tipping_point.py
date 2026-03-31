"""
AFD Feature: f_afd_bifurcation_tipping_point
靈感: 臨界分岔點 - 系統能量突變時的方向性觸發
"""
import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdBifurcationTippingPoint(BaseFeature):
    name = "f_afd_bifurcation_tipping_point"
    description = "臨界分岔點：系統能量突變時的方向性觸發，捕捉價格動能的臨界加速效應。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    BIN_SIZE = 200
    ZSCORE_WINDOW = 42

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Get raw tick data from kwargs
        tick_data = kwargs.get("_tick_raw")
        if tick_data is None or tick_data.empty:
            return pd.DataFrame(columns=["StockId", "Date", self.name])

        df = tick_data.copy()
        df["Date_int"] = df["Date"].astype(int)
        
        # Ensure sorting by TotalQty for deterministic ordering within same second
        df = df.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)

        results = []
        
        # Group by StockId and Date
        for (stock_id, date_int), df_day in df.groupby(["StockId", "Date_int"]):
            n = len(df_day)
            if n < self.BIN_SIZE:
                continue
                
            dc = df_day['DealCount'].values.astype(np.float64)
            pf = df_day['PrFlag'].values
            dp = df_day['DealPrice'].values

            n_bins = n // self.BIN_SIZE
            if n_bins < 2:
                continue

            energies = np.zeros(n_bins)
            dp_bins = np.zeros(n_bins)
            
            for i in range(n_bins):
                s, e = i * self.BIN_SIZE, (i + 1) * self.BIN_SIZE
                buy_f = np.sum(dc[s:e][pf[s:e] == 1]) / self.BIN_SIZE
                sell_f = np.sum(dc[s:e][pf[s:e] == 0]) / self.BIN_SIZE
                p_s = dp[s:e]
                x = np.arange(len(p_s), dtype=np.float64)
                
                if len(p_s) > 5:
                    sl5 = np.polyfit(x[:5], p_s[:5], 1)[0]
                    sl30 = np.polyfit(x[:min(30, len(x))], p_s[:min(30, len(p_s))], 1)[0]
                else:
                    sl5, sl30 = 0.0, 0.0

                energies[i] = buy_f ** 2 + sell_f ** 2 + sl5 ** 2 + sl30 ** 2
                dp_bins[i] = dp[e - 1] - dp[s]

            # Adaptive threshold: mean + 2σ
            threshold = np.mean(energies) + 2 * np.std(energies)
            # Sum of bins exceeding threshold with upward direction
            exceed = np.maximum(0, energies - threshold)
            dir_sign = np.sign(np.maximum(0, dp_bins))
            raw_val = np.sum(exceed * dir_sign)

            results.append({
                'StockId': stock_id,
                'Date': date_int,
                'raw': raw_val
            })

        if not results:
            return pd.DataFrame(columns=["StockId", "Date", self.name])
            
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(['StockId', 'Date']).reset_index(drop=True)
        
        # Apply rolling z-score
        df_res[self.name] = (
            zscore_rolling(df_res['raw'], window=self.ZSCORE_WINDOW, eps=1e-10)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-5, 5)
        )
        
        return df_res[["StockId", "Date", self.name]]
