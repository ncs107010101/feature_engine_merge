import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdLatentHeatPhaseLocking(BaseFeature):
    name = "f_afd_latent_heat_phase_locking"
    description = "潛熱相位鎖定：買賣單在時間序列上的瞬時同步性，捕捉賣單被秒吞噬的現象。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    BIN_SIZE = 30
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
            if len(df_day) < self.BIN_SIZE * 2:
                continue
                
            dc = df_day['DealCount'].values.astype(np.float64)
            pf = df_day['PrFlag'].values
            dp = df_day['DealPrice'].values

            buy_seq = np.where(pf == 1, dc, 0.0)
            sell_seq = np.where(pf == 0, dc, 0.0)

            n = len(dc)
            n_bins = n // self.BIN_SIZE
            if n_bins < 2:
                continue

            total_val = 0.0
            for i in range(n_bins):
                s, e = i * self.BIN_SIZE, (i + 1) * self.BIN_SIZE
                omega_low = buy_seq[s:e]
                omega_mid = sell_seq[s:e]
                inner_prod = np.dot(omega_low, omega_mid)
                p_slice = dp[s:e]
                x = np.arange(len(p_slice), dtype=np.float64)
                
                if np.std(x) > 0 and np.std(p_slice) > 0:
                    slope = np.polyfit(x, p_slice, 1)[0]
                else:
                    slope = 0.0
                trigger = np.sign(max(0.0, slope))
                total_val += inner_prod * trigger

            results.append({
                'StockId': stock_id,
                'Date': date_int,
                'raw': total_val
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
