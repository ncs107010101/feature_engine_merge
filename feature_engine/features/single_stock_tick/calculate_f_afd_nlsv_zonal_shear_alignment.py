import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdNlsvZonalShearAlignment(BaseFeature):
    name = "f_afd_nlsv_zonal_shear_alignment"
    description = "NLSV帶狀剪切對齊：大小單同向向上的瞬間協同，捕捉帶狀剪切效應。"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    BIN_SIZE = 100
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
            if n < self.BIN_SIZE:
                continue
                
            dc = df_day['DealCount'].values.astype(np.float64)
            pf = df_day['PrFlag'].values
            dp = df_day['DealPrice'].values

            q85 = np.percentile(dc, 85)
            q50 = np.percentile(dc, 50)
            large_mask = dc >= q85
            small_mask = dc <= q50
            direction = np.where(pf == 1, 1.0, np.where(pf == 0, -1.0, 0.0))

            n_bins = n // self.BIN_SIZE
            total_val = 0.0
            for i in range(n_bins):
                s, e = i * self.BIN_SIZE, (i + 1) * self.BIN_SIZE
                v_large = np.sum(dc[s:e][large_mask[s:e]] * direction[s:e][large_mask[s:e]])
                v_small = np.sum(dc[s:e][small_mask[s:e]] * direction[s:e][small_mask[s:e]])
                alignment = max(0, v_large * v_small)
                total_val += alignment

            dir_sign = np.sign(dp[-1] - dp[0])
            raw_val = total_val * dir_sign

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
