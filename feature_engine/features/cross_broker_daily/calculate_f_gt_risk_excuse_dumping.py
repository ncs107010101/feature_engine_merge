import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtRiskExcuseDumping(BaseFeature):
    name = "f_gt_risk_excuse_dumping"
    description = "風險藉口倒貨 - 當股價波動大時，前5大主力券商持續賣出"
    required_columns = ["StockId", "Date", "raw_top5_net_buy_nlargest", "raw_top5_net_sell_qtm", "收盤價", "最高價", "最低價", "開盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # Get top5 brokers by total_vol using lookback 60
        # For each stock, calculate rolling 60-day total volume per broker
        # This is approximated using available columns - we use top5_net_sell as proxy
        # Since we don't have per-broker data, we use the raw_top5_net_sell_qtm as top5 net sell
        # top5_net_sell = max(0, -(top5 NetBuy sum)) which equals raw_top5_net_buy_nlargest negated
        top5_net_sell = df["raw_top5_net_buy_nlargest"].apply(lambda x: max(0, -x))
        
        # high_risk = (high - low) / open
        high_risk = (df["最高價"] - df["最低價"]) / df["開盤價"].replace(0, np.nan)
        
        raw = top5_net_sell * high_risk
        
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
