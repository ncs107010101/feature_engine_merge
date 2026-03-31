import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdDiabaticConversionCross(BaseFeature):
    name = "f_afd_diabatic_conversion_cross"
    description = "非絕熱轉換交叉：主動單方向與價格位移的正交性，區分真假突破。"
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
            dc = df_day['DealCount'].values
            pf = df_day['PrFlag'].values

            V = np.where(pf == 1, dc, np.where(pf == 0, -dc, 0)).astype(np.float64)
            delta_p = np.diff(dp, prepend=dp[0])

            G_E = V * np.maximum(0, delta_p)
            C_A = V * np.abs(delta_p)

            raw_val = np.sum(G_E - C_A)

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
