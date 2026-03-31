import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdCondensationBurstAsymmetry(BaseFeature):
    name = "f_afd_condensation_burst_asymmetry"
    description = "凝結爆發非對稱：三次方運算子放大向上掃盤的極端非對稱方向。"
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
            dc = df_day['DealCount'].values.astype(np.float64)
            sp = df_day['SellPr'].values

            sell_pr_lag = np.roll(sp, 1)
            sell_pr_lag[0] = sp[0]
            C_t = np.maximum(0, dp - sell_pr_lag)

            burst = C_t * dc
            burst_cubed = burst ** 3
            raw_val = np.sum(burst_cubed)
            raw_val = np.log1p(np.abs(raw_val)) * np.sign(raw_val)

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
