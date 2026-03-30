import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FeatureGradualTrustEscalation(BaseFeature):
    name = "f_gt_gradual_trust_escalation"
    description = "Gradual Trust Escalation (消耗戰鎖碼): Top5 brokers by 20-day cum_net continue buying in falling market. Measures institutional conviction through sustained buying pressure despite adverse price movement."
    required_columns = [
        "StockId", "Date", 
        "top5_20d_netbuy", "top5_20d_positive_days", "price_stability"
    ]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # trust_test: top 5 brokers maintaining positive buying for 3+ days
        trust_test = (df["top5_20d_positive_days"] >= 3).astype(float)
        
        # escalation: current top5 netbuy / previous top5 netbuy
        prev_top5 = df.groupby("StockId")["top5_20d_netbuy"].shift(1)
        escalation = df["top5_20d_netbuy"] / (prev_top5 + 1e-8)
        escalation = escalation.fillna(1.0).replace([np.inf, -np.inf], 1.0)
        
        # price_ok: market not falling excessively
        price_ok = df["price_stability"].astype(float)
        
        # Raw signal
        raw = trust_test * escalation * price_ok
        
        # Rolling z-score normalization (span=20, using shift(1) for leakage prevention)
        def rolling_zscore(x, window=20):
            mean = x.shift(1).rolling(window=window, min_periods=5).mean()
            std = x.shift(1).rolling(window=window, min_periods=5).std()
            return (x - mean) / (std + 1e-8)
        
        out = rolling_zscore(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })