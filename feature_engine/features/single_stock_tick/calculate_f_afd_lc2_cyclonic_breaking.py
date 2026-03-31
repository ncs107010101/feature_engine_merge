import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdLc2CyclonicBreaking(BaseFeature):
    name = "f_afd_lc2_cyclonic_breaking"
    description = "LC2 氣旋碎波：買盤壓制賣盤的動能累積，捕捉強烈氣旋不對稱性。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    BIN_SIZE = 100
    ZSCORE_WINDOW = 20

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_data = kwargs.get("_tick_raw")
        if tick_data is None or tick_data.empty:
            return pd.DataFrame(columns=["StockId", "Date", self.name])

        df = tick_data.copy()
        df["Date_int"] = df["Date"].astype(int)
        df = df.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)

        results = []
        
        for (stock_id, date_int), df_day in df.groupby(["StockId", "Date_int"]):
            n = len(df_day)
            if n < self.BIN_SIZE:
                continue
                
            dc = df_day['DealCount'].values.astype(np.float64)
            pf = df_day['PrFlag'].values
            dp = df_day['DealPrice'].values

            n_bins = n // self.BIN_SIZE
            total_val = 0.0
            for i in range(n_bins):
                s, e = i * self.BIN_SIZE, (i + 1) * self.BIN_SIZE
                v_buy = np.sum(dc[s:e][pf[s:e] == 1])
                v_sell = np.sum(dc[s:e][pf[s:e] == 0])
                delta_v = max(0, v_buy - v_sell)
                delta_p = (dp[e - 1] - dp[s]) / (dp[s] + 1e-6)
                total_val += delta_v * delta_p

            results.append({
                'StockId': stock_id,
                'Date': date_int,
                'raw': total_val
            })

        if not results:
            return pd.DataFrame(columns=["StockId", "Date", self.name])
            
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(['StockId', 'Date']).reset_index(drop=True)
        df_res[self.name] = (
            zscore_rolling(df_res['raw'], window=self.ZSCORE_WINDOW, eps=1e-10)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-5, 5)
        )
        
        return df_res[["StockId", "Date", self.name]]
