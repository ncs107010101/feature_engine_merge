import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdOptimalPerturbationJacobian(BaseFeature):
    name = "f_afd_optimal_perturbation_jacobian"
    description = "最佳擾動Jacobian：2×2斜率矩陣的行列式，捕捉最大能量增幅。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    BIN_SIZE = 50
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
            if n < self.BIN_SIZE * 2:
                continue
                
            dp = df_day['DealPrice'].values.astype(np.float64)
            dc = df_day['DealCount'].values.astype(np.float64)

            n_bins = n // self.BIN_SIZE
            total_det = 0.0
            for i in range(n_bins):
                s, e = i * self.BIN_SIZE, (i + 1) * self.BIN_SIZE
                p_s, p_e = dp[s], dp[e - 1]
                v_s, v_e = np.sum(dc[s:s + 5]), np.sum(dc[e - 5:e])
                v_long_s, v_long_e = np.sum(dc[s:s + 30]), np.sum(dc[max(s, e - 30):e])

                slope_5p = (dp[min(s + 5, e - 1)] - dp[s]) / 5.0
                slope_30p = (dp[min(s + 30, e - 1)] - dp[s]) / 30.0
                slope_5v = (v_e - v_s) / (5.0 + 1e-10)
                slope_30v = (v_long_e - v_long_s) / (30.0 + 1e-10)

                det_s = slope_5p * slope_30v - slope_30p * slope_5v
                total_det += det_s

            dir_val = max(0, dp[-1] - dp[0])
            raw_val = total_det * dir_val

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
