import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdSvOptimizationTimeDivergence(BaseFeature):
    name = "f_afd_sv_optimization_time_divergence"
    description = "最佳化時間發散：短期推動力遠超長期時觸發，捕捉最佳擾動敏感性。"
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

            delta_p = np.diff(dp, prepend=dp[0])

            alpha_s = 2.0 / (5 + 1)
            alpha_l = 2.0 / (60 + 1)
            v_short = np.zeros(n, dtype=np.float64)
            v_long = np.zeros(n, dtype=np.float64)
            v_short[0] = delta_p[0]
            v_long[0] = delta_p[0]
            for i in range(1, n):
                v_short[i] = alpha_s * delta_p[i] + (1 - alpha_s) * v_short[i - 1]
                v_long[i] = alpha_l * delta_p[i] + (1 - alpha_l) * v_long[i - 1]

            divergence = v_short - v_long
            weighted = np.maximum(0, divergence * dc)
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
