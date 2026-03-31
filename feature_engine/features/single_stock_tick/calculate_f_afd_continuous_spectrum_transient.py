import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdContinuousSpectrumTransient(BaseFeature):
    name = "f_afd_continuous_spectrum_transient"
    description = "連續譜暫態：超高頻與長週期趨勢的瞬間對齊，捕捉非模態增長。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

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
            if n < 65:
                continue
                
            dp = df_day['DealPrice'].values.astype(np.float64)
            dc = df_day['DealCount'].values.astype(np.float64)

            s_cont = np.diff(dp, n=1, prepend=dp[0])
            s_disc = np.zeros(n, dtype=np.float64)
            s_disc[60:] = dp[60:] - dp[:-60]

            alignment = np.maximum(0, s_cont * s_disc)
            weighted = alignment * dc
            raw_val = np.sum(weighted)

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
