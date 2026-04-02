import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureGtReputationManipulationFraud(BaseFeature):
    name = "f_gt_reputation_manipulation_fraud"
    description = "small_buy_ticks intensity × top5_net_sell dump magnitude, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_small_buy_ticks", "raw_top5_net_sell_qtm"]
    data_combination = "cross_tick_broker_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        df["raw_small_buy_ticks"] = df["raw_small_buy_ticks"].fillna(0)
        df["raw_top5_net_sell_qtm"] = df["raw_top5_net_sell_qtm"].fillna(0)
        
        sbt_mean = df["raw_small_buy_ticks"].groupby(df["StockId"]).transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        fake_intensity = df["raw_small_buy_ticks"] / (sbt_mean + 1e-5)
        
        ns_mean = df["raw_top5_net_sell_qtm"].groupby(df["StockId"]).transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        dump_mag = df["raw_top5_net_sell_qtm"] / (ns_mean + 1e-5)
        dump_mag = dump_mag.fillna(0)
        dump_mag = np.where(df["raw_top5_net_sell_qtm"] <= 0, 0, dump_mag)
        
        raw = fake_intensity * dump_mag
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
