import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ewm_smooth, ts_rank_center

@register_feature
class FVolbarOfiSkewEwm20TsrankEwm20(BaseFeature):
    name = "f_volbar_ofi_skew_ewm20_tsrank_ewm20"
    description = "Tick feature: f_volbar_ofi_skew_ewm20_tsrank_ewm20"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)


        out_series = g["raw_ofi_skew"].transform(lambda x: x.ewm(span=20, min_periods=1).mean()).groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120)).groupby(df["StockId"]).transform(lambda x: x.ewm(span=20, min_periods=1).mean())
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
