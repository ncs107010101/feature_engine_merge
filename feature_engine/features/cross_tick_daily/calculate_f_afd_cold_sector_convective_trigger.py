import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdColdSectorConvectiveTrigger(BaseFeature):
    name = "f_afd_cold_sector_convective_trigger"
    description = "冷區對流加熱: 冷區主動買量 × 冷區斜率, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_cold_zone_buy", "raw_cold_slope", "vwap"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 20
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_cold_zone_buy"] * df["raw_cold_slope"]
        out = raw.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, self.ZSCORE_WINDOW, eps=1e-10)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
