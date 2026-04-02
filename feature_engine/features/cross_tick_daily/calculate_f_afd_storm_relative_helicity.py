import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdStormRelativeHelicity(BaseFeature):
    name = "f_afd_storm_relative_helicity"
    description = "風暴相對螺旋度: (tick_mean_price - vwap) × 5日報酬率滾動和, rolling_zscore(42)"
    required_columns = ["StockId", "Date", "raw_tick_mean_price", "vwap", "報酬率"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 42
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        v_srf = df["raw_tick_mean_price"].values - df["vwap"].values
        s_env = df.groupby("StockId")["報酬率"].transform(
            lambda x: x.rolling(5).sum().fillna(0)
        ).values
        
        cross = np.abs(v_srf * s_env)
        dot = v_srf * s_env
        raw = cross * np.sign(dot)
        raw = pd.Series(raw)
        
        out = raw.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, self.ZSCORE_WINDOW, eps=1e-10)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
