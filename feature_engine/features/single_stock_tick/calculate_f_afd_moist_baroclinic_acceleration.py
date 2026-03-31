import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdMoistBaroclinicAcceleration(BaseFeature):
    name = "f_afd_moist_baroclinic_acceleration"
    description = "濕斜壓加速度：凝結潛熱使線性成長轉化為指數級爆發，捕捉成長率的二階加速度。"
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

            m_moist = (dp[5:] - dp[:-5]) / 5.0
            m_dry_full = np.zeros(n, dtype=np.float64)
            m_dry_full[60:] = (dp[60:] - dp[:-60]) / 60.0
            m_dry = m_dry_full[5:]

            delta_p = np.maximum(0, np.diff(dp, prepend=dp[0]))
            delta_p_aligned = delta_p[5:]

            accel = (m_moist - m_dry) * delta_p_aligned
            raw_val = np.sum(accel)

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
