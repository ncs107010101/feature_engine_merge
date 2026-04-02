import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdNlsvMeridionalExtensionTensor(BaseFeature):
    name = "f_afd_nlsv_meridional_extension_tensor"
    description = "NLSV經向延展張量：時空域張量跡，捕捉波干擾防止效應。"
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
            if n < 10:
                continue
                
            dp = df_day['DealPrice'].values.astype(np.float64)
            ts = df_day['DealTimeSecond'].values.astype(np.float64)

            t_flow = np.diff(ts, prepend=ts[0]).astype(np.float64)
            p_dist = np.diff(dp, prepend=dp[0])

            trace_val = t_flow * p_dist
            weighted = trace_val * np.maximum(0, p_dist)
            raw_val = np.sum(weighted)

            results.append({
                'StockId': stock_id,
                'Date': date_int,
                'raw': np.log1p(np.abs(raw_val)) * np.sign(raw_val)
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
