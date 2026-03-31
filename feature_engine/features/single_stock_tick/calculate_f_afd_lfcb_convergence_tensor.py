import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdLfcbConvergenceTensor(BaseFeature):
    name = "f_afd_lfcb_convergence_tensor"
    description = "邊界水平輻合張量：買賣推進梯度的張量跡，捕捉垂直渦度拉伸。"
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
            if n < self.BIN_SIZE * 2:
                continue
                
            dc = df_day['DealCount'].values.astype(np.float64)
            pf = df_day['PrFlag'].values
            dp = df_day['DealPrice'].values

            n_bins = n // self.BIN_SIZE
            buy_totals = np.zeros(n_bins)
            sell_totals = np.zeros(n_bins)
            for i in range(n_bins):
                s, e = i * self.BIN_SIZE, (i + 1) * self.BIN_SIZE
                buy_totals[i] = np.sum(dc[s:e][pf[s:e] == 1])
                sell_totals[i] = np.sum(dc[s:e][pf[s:e] == 0])

            buy_grad = np.diff(buy_totals, prepend=buy_totals[0])
            sell_grad = np.diff(sell_totals, prepend=sell_totals[0])

            trace_c = buy_grad + sell_grad
            total_trace = np.sum(trace_c)

            dir_val = max(0, dp[-1] - dp[0])
            raw_val = total_trace * dir_val

            results.append({
                'StockId': stock_id,
                'Date': date_int,
                'raw': raw_val
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
