import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdOcclusionZoneVorticity(BaseFeature):
    name = "f_afd_occlusion_zone_vorticity"
    description = "錮囚區渦度: (廣大買盤支撐 - 高價區買盤) × 價格變動, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "above_vwap_buy", "above_vwap_high_buy", "close_pr", "open_pr"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 20
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        f_broad = df["above_vwap_buy"].values.astype(np.float64)
        f_narrow = df["above_vwap_high_buy"].values.astype(np.float64)
        delta_close = df["close_pr"].values - df["open_pr"].values
        
        raw = np.maximum(0, f_broad - f_narrow) * delta_close
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
