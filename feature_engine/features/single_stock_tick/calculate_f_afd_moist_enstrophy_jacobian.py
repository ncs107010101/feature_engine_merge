import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdMoistEnstrophyJacobian(BaseFeature):
    name = "f_afd_moist_enstrophy_jacobian"
    description = "濕度擬能Jacobian：Jacobian行列式衡量相空間收縮/擴張，捕捉非線性演化。"
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
            if len(df_day) < self.BIN_SIZE * 2:
                continue
                
            dp = df_day['DealPrice'].values
            tq = df_day['TotalQty'].values.astype(np.float64)
            ts = df_day['DealTimeSecond'].values.astype(np.float64)

            n = len(dp)
            n_bins = n // self.BIN_SIZE
            if n_bins < 2:
                continue

            total_det = 0.0
            for i in range(n_bins):
                s, e = i * self.BIN_SIZE, (i + 1) * self.BIN_SIZE
                p_s, p_e = dp[s], dp[e - 1]
                q_s, q_e = tq[s], tq[e - 1]
                t_s, t_e = ts[s], ts[e - 1]

                dt = t_e - t_s + 1e-6
                dq = q_e - q_s + 1e-6
                d_price = p_e - p_s

                dp_dt = d_price / dt
                dp_dq = d_price / dq
                dq_dt = (q_e - q_s) / dt

                det_j = dp_dt - dp_dq * dq_dt

                if dp_dt > 0:
                    total_det += det_j

            results.append({
                'StockId': stock_id,
                'Date': date_int,
                'raw': total_det
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
