import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdVorticityRiverTiltingJacobian(BaseFeature):
    name = "f_afd_vorticity_river_tilting_jacobian"
    description = "渦度河流傾斜Jacobian：量價流形拉伸的Jacobian行列式。"
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
            ts = df_day['DealTimeSecond'].values.astype(np.float64)

            n_bins = n // self.BIN_SIZE
            total_det = 0.0
            for i in range(n_bins):
                s, e = i * self.BIN_SIZE, (i + 1) * self.BIN_SIZE
                dv = np.sum(dc[s:e]) - np.sum(dc[max(0, s - self.BIN_SIZE):s]) if i > 0 else np.sum(dc[s:e])
                d_price = dp[e - 1] - dp[s]
                dt = ts[e - 1] - ts[s] + 1e-6

                dv_dt = dv / dt
                dp_dt = d_price / dt
                dv_dp = dv / (d_price + 1e-8)
                det_j = dv_dt - dv_dp * dp_dt
                total_det += det_j

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
