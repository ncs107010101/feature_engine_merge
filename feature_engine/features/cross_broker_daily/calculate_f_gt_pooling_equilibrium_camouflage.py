import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePoolingEquilibriumCamouflage(BaseFeature):
    name = "f_gt_pooling_equilibrium_camouflage"
    description = "Pooling Equilibrium Camouflage (Pool均衡偽裝): Volume surge + low concentration."
    required_columns = [
        "StockId", "Date",
        "raw_hhi_buy", "vol_surge"
    ]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # Low concentration: HHI below 20th percentile of its own 20-day history
        hhi_roll_q20 = df.groupby("StockId")["raw_hhi_buy"].transform(
            lambda x: x.shift(1).rolling(window=20, min_periods=5).quantile(0.2)
        )
        low_concentration = (df["raw_hhi_buy"] < hhi_roll_q20).astype(float)
        
        # Signal: vol_surge AND low concentration
        raw = df["vol_surge"].astype(float) * low_concentration
        
        # EWM then z-score
        def ewm_then_zscore(x, span=20):
            ewm = x.shift(1).ewm(span=span, adjust=False).mean()
            mean = ewm.rolling(window=span, min_periods=5).mean()
            std = ewm.rolling(window=span, min_periods=5).std()
            return (ewm - mean) / (std + 1e-8)
        
        out = ewm_then_zscore(pd.Series(raw.values), span=5)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })