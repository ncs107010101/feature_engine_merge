import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdPvPtResonanceJacobian(BaseFeature):
    name = "f_afd_pv_pt_resonance_jacobian"
    description = "宏觀(VWAP)與微觀(tick)斜率的共振 Jacobian: det(M) × max(0, close_p - vwap), rolling_zscore(42)"
    required_columns = ["StockId", "Date", "vwap", "tick_slope", "close_pr"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 42
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        vwap = df["vwap"].values
        tick_slope = df["tick_slope"].values
        close_p = df["close_pr"].values
        
        vwap_slope = np.zeros_like(vwap)
        vwap_slope[5:] = (vwap[5:] - vwap[:-5]) / (vwap[:-5] + 1e-8)
        
        det_m = vwap_slope * tick_slope**2 - vwap_slope**2 * tick_slope
        dir_val = np.maximum(0, close_p - vwap)
        raw = det_m * dir_val
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
