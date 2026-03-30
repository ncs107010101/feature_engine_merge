import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdPvUnshieldingOrr(BaseFeature):
    name = "f_afd_pv_unshielding_orr"
    description = "AFD特徵：位渦去屏蔽Orr機制"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            tick_raw = data
        
        df = tick_raw.copy()
        df["Date"] = df["Date"].astype(int)
        df = df.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
        df["Date_int"] = df["Date"]
        
        df["_direction"] = np.where(df["PrFlag"] == 1, 1.0, np.where(df["PrFlag"] == 0, -1.0, 0.0))
        
        def compute_orr(group):
            direction = group["_direction"].values
            
            if len(direction) < 20:
                return 0.0
            
            corr_sum = 0.0
            count = 0
            for lag in range(1, min(10, len(direction) - 1)):
                c1 = direction[:-lag]
                c2 = direction[lag:]
                if len(c1) > 0 and len(c2) > 0:
                    mean1, mean2 = c1.mean(), c2.mean()
                    std1, std2 = c1.std(), c2.std()
                    if std1 > 0 and std2 > 0:
                        corr = np.corrcoef(c1, c2)[0, 1]
                        if not np.isnan(corr):
                            corr_sum += corr
                            count += 1
            
            return corr_sum / max(count, 1)
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_orr).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
