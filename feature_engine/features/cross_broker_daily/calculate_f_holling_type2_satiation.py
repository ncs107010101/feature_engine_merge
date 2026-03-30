import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FHollingType2Satiation(BaseFeature):
    name = "f_holling_type2_satiation"
    description = "掠食者飽食邊界 - Top5賣出量的二階導數，只在短期創新低日計算"
    required_columns = ["StockId", "Date", "raw_is_new_low_5d", "raw_top5_sell"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        top5_sell_d1 = df["raw_top5_sell"].diff().fillna(0)
        top5_sell_d2 = top5_sell_d1.diff().fillna(0)
        
        raw = np.where(
            df["raw_is_new_low_5d"] == 1.0,
            top5_sell_d2,
            0.0
        )
        
        raw_series = pd.Series(raw, index=df.index)
        
        raw_ewm = g["raw_top5_sell"].transform(
            lambda x: raw_series.loc[x.index].ewm(span=10, min_periods=1).mean()
        )
        
        out_series = g["raw_is_new_low_5d"].transform(lambda x: zscore_rolling(raw_ewm.loc[x.index], 42))
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
