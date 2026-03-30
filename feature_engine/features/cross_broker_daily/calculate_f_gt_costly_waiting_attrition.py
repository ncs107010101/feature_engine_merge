import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtCostlyWaitingAttrition(BaseFeature):
    name = "f_gt_costly_waiting_attrition"
    description = "消耗戰鎖碼：股價連跌時前5大主力仍持續淨買，滾動5日加總×是否下跌作為 raw，zscore(20)標準化。遊戲理論中展示承受虧損的決心。"
    required_columns = ["StockId", "Date", "raw_top5_net_buy_nlargest", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        ret5 = df.groupby("StockId")["收盤價"].pct_change(5)
        is_falling = (ret5 < 0).astype(float)
        
        top5_rolling = df.groupby("StockId")["raw_top5_net_buy_nlargest"].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )
        raw = top5_rolling * is_falling
        
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
