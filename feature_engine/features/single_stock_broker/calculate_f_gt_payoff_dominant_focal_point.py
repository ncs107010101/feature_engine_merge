import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FGtPayoffDominantFocalPoint(BaseFeature):
    name = "f_gt_payoff_dominant_focal_point"
    description = "Game Theory v17: Focal point power = distinct brokers × buy volume at round prices (Price % 10 == 0). Positive → Market consensus at round prices → Extreme HIGH return"
    required_columns = ["StockId", "Date", "raw_focal_distinct_brokers", "raw_focal_buy_vol"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        focal_power = df["raw_focal_distinct_brokers"] * df["raw_focal_buy_vol"]
        ewm = focal_power.groupby(df["StockId"]).transform(lambda x: x.ewm(span=5, min_periods=1).mean())
        out = ewm.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
