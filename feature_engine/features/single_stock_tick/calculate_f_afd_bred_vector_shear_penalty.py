import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdBredVectorShearPenalty(BaseFeature):
    name = "f_afd_bred_vector_shear_penalty"
    description = "繁殖向量剪切 penalty：前段趨勢與後段擾動的幾何交叉，捕捉洗盤後轉折。"
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
            if n < 20:
                continue
                
            dc = df_day['DealCount'].values.astype(np.float64)
            pf = df_day['PrFlag'].values
            dp = df_day['DealPrice'].values

            split = int(n * 0.6)
            direction = np.where(pf == 1, 1.0, np.where(pf == 0, -1.0, 0.0))

            h_flow = np.sum(dc[:split] * direction[:split])
            h_price = dp[split - 1] - dp[0]
            n_flow = np.sum(dc[split:] * direction[split:])
            n_price = dp[-1] - dp[split]

            dot = h_flow * n_flow + h_price * n_price
            cross = abs(h_flow * n_price - h_price * n_flow)

            dir_sign = np.sign(dp[-1] - dp[0])
            raw_val = cross * min(0, dot) * dir_sign

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
