import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureBeSafeEnvironmentExitSpite(BaseFeature):
    name = "f_be_safe_environment_exit_spite"
    description = "safe_env × failed_breakout × (pm_small_sell / total_vol), rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_last500_small_sell", "raw_vol20", "raw_vol20_q20", "raw_high_shift1", "total_vol", "最高價"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        df["raw_vol20"] = df["raw_vol20"].fillna(0)
        df["raw_vol20_q20"] = df["raw_vol20_q20"].fillna(0)
        df["raw_high_shift1"] = df["raw_high_shift1"].fillna(0)
        df["raw_last500_small_sell"] = df["raw_last500_small_sell"].fillna(0)
        df["total_vol"] = df["total_vol"].fillna(0)
        
        safe_env = (df["raw_vol20"] < df["raw_vol20_q20"]).astype(int)
        
        prev_high = df.groupby("StockId")["raw_high_shift1"].shift(0)
        current_high = df["最高價"].astype(float)
        failed_breakout = (current_high < prev_high).fillna(0).astype(int)
        
        spite = df["raw_last500_small_sell"] / (df["total_vol"] + 1e-5)
        
        raw = safe_env * failed_breakout * spite
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
