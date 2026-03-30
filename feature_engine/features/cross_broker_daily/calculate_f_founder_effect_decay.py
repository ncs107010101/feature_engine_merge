import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FFounderEffectDecay(BaseFeature):
    name = "f_founder_effect_decay"
    description = "創始者效應衰退 - Top5買超券商從建倉到當日累積未平倉佔最初建倉量的比例衰減程度"
    required_columns = ["StockId", "Date", "raw_is_founder_day", "raw_top5_net_buy"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        decay_vals = np.zeros(len(df))
        initial_buy = np.zeros(len(df))
        current_inv = np.zeros(len(df))
        
        for sid in df["StockId"].unique():
            mask = df["StockId"] == sid
            idx = df[mask].index
            
            inv = 0.0
            init = 0.0
            
            for pos in idx:
                is_founder = df.loc[pos, "raw_is_founder_day"] == 1.0
                top5_net = df.loc[pos, "raw_top5_net_buy"]
                
                if is_founder and top5_net > 0:
                    init = top5_net
                    inv = top5_net
                else:
                    inv = max(inv + top5_net, 0.0)
                
                if init > 0:
                    decay_vals[pos] = inv / init
                else:
                    decay_vals[pos] = 0.0
                
                initial_buy[pos] = init
                current_inv[pos] = inv
        
        decay_series = pd.Series(decay_vals, index=df.index)
        raw_diff = -decay_series.diff().fillna(0.0)
        
        out_series = raw_diff.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, 42)
        )
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
