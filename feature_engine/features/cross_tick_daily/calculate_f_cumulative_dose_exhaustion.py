import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import rolling_zscore


@register_feature
class FeatureCumulativeDoseExhaustion(BaseFeature):
    name = "f_cumulative_dose_exhaustion"
    description = "累積劑量枯竭點 (買盤衰竭)。當連續上漲天數越多且均量越大，但當日主動賣量相對小時，代表買盤即將衰竭。"
    required_columns = ["StockId", "Date", "raw_consecutive_up_days", "raw_consecutive_up_vol", "raw_active_sell"]
    data_combination = "cross_tick_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        raw = (df["raw_consecutive_up_days"] * df["raw_consecutive_up_vol"]) / (df["raw_active_sell"] + 1e-8)
        
        out = raw.groupby(df["StockId"]).transform(lambda x: rolling_zscore(x, 42))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
