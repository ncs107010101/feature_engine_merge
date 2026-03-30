import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtEchoChamberAssortative(BaseFeature):
    name = "f_gt_echo_chamber_assortative"
    description = "同溫層泡泡：成交量放大但前10大主力參與率下降，代表散戶對敲泡沫。ewm(5)→zscore(20)標準化。"
    required_columns = ["StockId", "Date", "raw_top10_total_vol", "raw_total_broker_vol", "成交量(千股)"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        vol = df["成交量(千股)"].fillna(0)
        vol_ma20 = vol.rolling(20, min_periods=1).mean().shift(1)
        vol_surge = vol / (vol_ma20 + 1e-8)
        
        top10_ratio = df["raw_top10_total_vol"] / (df["raw_total_broker_vol"] + 1e-8)
        
        raw = vol_surge * (1 - top10_ratio)
        raw_ewm = raw.ewm(span=5, min_periods=1).mean()
        
        out = zscore_rolling(raw_ewm, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
