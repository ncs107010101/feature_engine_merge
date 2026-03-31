import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdWaveMeanFlowJacobian(BaseFeature):
    name = "f_afd_wave_mean_flow_jacobian"
    description = "波與平均流Jacobian：5分鐘/30分鐘尺度的Jacobian行列式，捕捉短波動能反饋。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    SEC_PER_BIN = 300
    ZSCORE_WINDOW = 42

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
            if n < 50:
                continue
                
            dp = df_day['DealPrice'].values
            ts = df_day['DealTimeSecond'].values

            bin_start = 32400
            bin_end = 48600
            bins_edges = np.arange(bin_start, bin_end + self.SEC_PER_BIN, self.SEC_PER_BIN)
            bin_idx = np.digitize(ts, bins_edges) - 1

            slopes_5min = []
            for b in range(len(bins_edges) - 1):
                mask = bin_idx == b
                if np.sum(mask) < 3:
                    slopes_5min.append(0.0)
                    continue
                p_bin = dp[mask]
                x = np.arange(len(p_bin), dtype=np.float64)
                slope = np.polyfit(x, p_bin, 1)[0]
                slopes_5min.append(slope)

            slopes_5min = np.array(slopes_5min)
            if len(slopes_5min) < 7:
                results.append({'StockId': stock_id, 'Date': date_int, 'raw': 0.0})
                continue

            slopes_30min = np.convolve(slopes_5min, np.ones(6) / 6, mode='valid')

            min_len = min(len(slopes_5min) - 1, len(slopes_30min) - 1)
            if min_len < 2:
                results.append({'StockId': stock_id, 'Date': date_int, 'raw': 0.0})
                continue

            total_det = 0.0
            for i in range(min_len):
                det_m = slopes_5min[i] * slopes_30min[i + 1] - slopes_5min[i + 1] * slopes_30min[i]
                total_det += det_m

            close_p = dp[-1]
            open_p = dp[0]
            dir_sign = np.sign(max(0, close_p - open_p))
            raw_val = total_det * dir_sign

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
