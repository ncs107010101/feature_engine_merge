import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, ts_rank_center

@register_feature
class FIntradayMomentumDecay(BaseFeature):
    name = "f_intraday_momentum_decay"
    description = "Tick feature: f_intraday_momentum_decay"
    required_columns = ["StockId", "Date", "raw_intraday_momentum_decay"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["raw_intraday_momentum_decay"].transform(lambda x: ewm_smooth(x.fillna(0), 5))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
