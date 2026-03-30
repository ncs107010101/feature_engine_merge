import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore


@register_feature
class FeatureGtCharacterStigmaAvoidance(BaseFeature):
    name = "f_gt_character_stigma_avoidance"
    description = "品格污名迴避 - 頂級賣家出貨但掩蓋痕跡"
    required_columns = ["StockId", "Date", "raw_top5_net_sell_qtm", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # sellers sorted by NetSell descending
        # r1, r2, r3 = top 3 sellers' NetSell
        # Using raw_top5_net_sell_qtm to approximate top seller
        r1 = df["raw_top5_net_sell_qtm"]
        r2 = df["raw_top5_net_sell_qtm"] * 0.7
        r3 = df["raw_top5_net_sell_qtm"] * 0.5
        
        stigma_gap = (r1 - r2) + (r1 - r3)
        
        r1_mean = df.groupby("StockId")["raw_top5_net_sell_qtm"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        heavy = (r1 > r1_mean * 2).astype(int)
        
        raw = heavy / (stigma_gap + 1e-5)
        
        out = ewm_then_zscore(raw, ewm_span=5, z_window=20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
