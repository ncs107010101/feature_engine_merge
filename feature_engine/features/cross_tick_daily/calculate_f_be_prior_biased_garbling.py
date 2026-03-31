import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureBePriorBiasedGarbling(BaseFeature):
    name = "f_be_prior_biased_garbling"
    description = "bull_prior × garbling × dumping, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "raw_bin200_garbling_dumping", "raw_ret5"]
    data_combination = "cross_tick_daily"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        df["raw_bin200_garbling_dumping"] = df["raw_bin200_garbling_dumping"].fillna(0)
        df["raw_ret5"] = df["raw_ret5"].fillna(0)
        
        strong_bull = (df["raw_ret5"] > 0.10).astype(int)
        
        raw = strong_bull * df["raw_bin200_garbling_dumping"]
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
