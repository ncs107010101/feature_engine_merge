import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdLc1FrontalGravityWave(BaseFeature):
    name = "f_afd_lc1_frontal_gravity_wave"
    description = "LC1 前鋒重力波：掛單壓力與價格位移的正交程度，捕捉冷鋒激發效應。"
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
                
            dp = df_day['DealPrice'].values
            sp = df_day['SellPr'].values
            bp = df_day['BuyPr'].values

            spread = sp - bp
            grad_lob = np.diff(spread, prepend=spread[0])
            delta_p = np.diff(dp, prepend=dp[0])

            cross_vals = np.abs(
                grad_lob[:-1] * delta_p[1:] - delta_p[:-1] * grad_lob[1:]
            )
            sign_dp = np.sign(delta_p[:-1])
            weighted_cross = cross_vals * sign_dp

            raw_val = np.sum(weighted_cross)

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
