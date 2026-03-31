import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdLatentHeatWaterfallSlope(BaseFeature):
    name = "f_afd_latent_heat_waterfall_slope"
    description = "融券燃料點火: 融券 × 正向tick斜率 (conditional trigger), rolling_zscore(42)"
    required_columns = ["StockId", "Date", "raw_margin_balance", "tick_slope", "vwap"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 42
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        s_short = df["raw_margin_balance"].fillna(0).values.astype(np.float64)
        p_slope = df["tick_slope"].values
        trigger = (s_short > 0) & (p_slope > 0)
        raw = pd.Series(np.where(trigger, s_short * p_slope, 0.0))
        
        out = raw.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, self.ZSCORE_WINDOW, eps=1e-10)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
