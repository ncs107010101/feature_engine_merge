import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdStreamfunctionVarianceFilter(BaseFeature):
    name = "f_afd_streamfunction_variance_filter"
    description = "流函數方差濾波：大單正向推升與碎單噪聲的信雜比。"
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
                
            dp = df_day['DealPrice'].values.astype(np.float64)
            dc = df_day['DealCount'].values.astype(np.float64)

            q85 = np.percentile(dc, 85)
            q50 = np.percentile(dc, 50)

            delta_p = np.diff(dp, prepend=dp[0])
            large_mask = dc >= q85
            small_mask = dc <= q50

            signal = np.sum(np.maximum(0, delta_p[large_mask]) * dc[large_mask])
            noise = np.sum(np.abs(delta_p[small_mask] * dc[small_mask])) + 1e-10

            raw_val = signal / noise

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
