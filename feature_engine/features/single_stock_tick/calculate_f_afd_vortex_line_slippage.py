import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdVortexLineSlippage(BaseFeature):
    name = "f_afd_vortex_line_slippage"
    description = "渦線滑移：早盤砸盤→中盤吸籌→尾盤拉升的三因子捕捉。"
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
            if n < 30:
                continue
                
            dp = df_day['DealPrice'].values.astype(np.float64)
            dc = df_day['DealCount'].values.astype(np.float64)
            pf = df_day['PrFlag'].values

            third = n // 3
            early_ret = dp[third - 1] - dp[0]
            early_sell = np.sum(dc[:third][pf[:third] == 0])
            D = min(0, early_ret) * early_sell

            mid_buy = np.sum(dc[third:2 * third][pf[third:2 * third] == 1])
            omega = mid_buy

            late_ret = dp[-1] - dp[2 * third]
            late_buy = np.sum(dc[2 * third:][pf[2 * third:] == 1])
            U = max(0, late_ret) * late_buy

            raw_val = omega * abs(D) * U
            raw_val = np.log1p(raw_val)

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
