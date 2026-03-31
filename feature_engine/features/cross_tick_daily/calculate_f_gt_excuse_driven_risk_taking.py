import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtExcuseDrivenRiskTaking(BaseFeature):
    name = "f_gt_excuse_driven_risk_taking"
    description = "loss_domain × volatility × sell_vol 日最大值, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_loss_domain_vol_max"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = df["raw_loss_domain_vol_max"].fillna(0)
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
