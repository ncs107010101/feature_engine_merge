import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import ts_rank_center


@register_feature
class FReversalComposite(BaseFeature):
    name = "f_reversal_composite"
    description = "Reversal composite (time-series normalized)"
    required_columns = ["StockId", "Date", "_intra_ret", "開盤價", "_prev_close", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        f_open_gap = (df["開盤價"] - df["_prev_close"]) / (df["_prev_close"] + 1e-9)
        f_ret_5d = g["收盤價"].pct_change(5)
        
        # Use time-series rank instead of cross-sectional rank
        r_intra = df["_intra_ret"].groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        r_gap = f_open_gap.groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        r_5d = f_ret_5d.groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        
        # Negate: higher recent returns = more reversal potential (contrarian)
        out_series = -(r_intra * 0.30 + r_gap * 0.30 + r_5d * 0.40)
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=-1, upper=1)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
