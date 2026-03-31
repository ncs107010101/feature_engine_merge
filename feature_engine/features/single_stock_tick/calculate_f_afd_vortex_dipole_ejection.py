import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdVortexDipoleEjection(BaseFeature):
    name = "f_afd_vortex_dipole_ejection"
    description = "渦旋偶極子彈射：早盤/尾盤流動張量的正交性，捕捉日內動能轉換。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

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
            if n < 20:
                continue
                
            dc = df_day['DealCount'].values.astype(np.float64)
            pf = df_day['PrFlag'].values
            dp = df_day['DealPrice'].values

            mid = n // 2
            am_buy = np.sum(dc[:mid][pf[:mid] == 1])
            am_sell = np.sum(dc[:mid][pf[:mid] == 0])
            pm_buy = np.sum(dc[mid:][pf[mid:] == 1])
            pm_sell = np.sum(dc[mid:][pf[mid:] == 0])

            trace_val = am_buy * pm_buy + am_sell * pm_sell
            dir_val = max(0, dp[-1] - dp[0])
            raw_val = trace_val * dir_val

            results.append({
                'StockId': stock_id,
                'Date': date_int,
                'raw': np.log1p(raw_val)
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
