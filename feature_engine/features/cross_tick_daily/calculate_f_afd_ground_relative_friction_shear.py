import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdGroundRelativeFrictionShear(BaseFeature):
    name = "f_afd_ground_relative_friction_shear"
    description = "大盤摩擦剪切: 超額報酬 × 大盤(20日滾動平均) × 成交量, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "報酬率", "total_vol"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 20
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        stock_ret = df["報酬率"].fillna(0).values
        market_ret = df.groupby("StockId")["報酬率"].transform(
            lambda x: x.rolling(20).mean().fillna(0)
        ).values
        v_micro = np.maximum(0, stock_ret - market_ret)
        vol = df["total_vol"].values.astype(np.float64)
        
        raw = market_ret * v_micro * vol
        raw = np.log1p(np.abs(raw)) * np.sign(raw)
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
