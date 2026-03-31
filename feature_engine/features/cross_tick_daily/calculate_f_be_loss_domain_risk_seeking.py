import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureBeLossDomainRiskSeeking(BaseFeature):
    name = "f_be_loss_domain_risk_seeking"
    description = "處於虧損區間(VWAP<開盤*0.98) × (小單成交價標準差/均價) 的日最大值, ewm_then_zscore(5, 20)"
    required_columns = ["StockId", "Date", "raw_loss_domain_max"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_loss_domain_max"]
        smoothed = raw.groupby(df["StockId"]).transform(lambda x: x.ewm(span=5, min_periods=5).mean())
        out = smoothed.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })