import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtSecondOrderBeliefFading(BaseFeature):
    name = "f_gt_second_order_belief_fading"
    description = "二階信念褪散：股價突破20日新高時，前5大主力趁機淨賣出收割。is_breakout×Top5_NetSell作 raw，zscore(60)標準化。"
    required_columns = ["StockId", "Date", "raw_top5_net_sell_qtm", "收盤價", "最高價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        rolling_high_20 = df.groupby("StockId")["最高價"].transform(
            lambda x: x.shift(1).rolling(20, min_periods=1).max()
        )
        is_breakout = (df["收盤價"] > rolling_high_20).astype(float)
        
        raw = is_breakout * df["raw_top5_net_sell_qtm"].abs()
        
        out = zscore_rolling(raw, 60)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
