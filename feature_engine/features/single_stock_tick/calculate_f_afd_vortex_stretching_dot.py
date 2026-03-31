import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdVortexStretchingDot(BaseFeature):
    name = "f_afd_vortex_stretching_dot"
    description = "渦度拉伸：向上價格與流動性真空梯度的內積。"
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
            if len(df_day) < 10:
                continue
                
            dp = df_day['DealPrice'].values
            bp = df_day['BuyPr'].values
            sp = df_day['SellPr'].values

            spread = sp - bp
            grad_liq = np.diff(spread, prepend=spread[0])
            v = np.diff(dp, prepend=dp[0])

            trigger = (v > 0).astype(np.float64)
            stretching = v * grad_liq * trigger

            raw_val = np.sum(stretching)

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
