import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import ts_rank_center


@register_feature
class FHlRangeEwm(BaseFeature):
    name = "f_hl_range_ewm"
    description = "HL range ewm (time-series normalized)"
    required_columns = ["StockId", "Date", "最高價", "最低價", "收盤價"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        _hl_raw = (df["最高價"] - df["最低價"]) / (df["收盤價"] + 1e-9)
        _hl_ewm = _hl_raw.groupby(df["StockId"]).transform(lambda x: x.ewm(span=5, min_periods=3).mean())
        _hl_log = np.log1p(_hl_ewm)
        
        # Use time-series rank instead of cross-sectional rank
        # Negate: lower HL range = less volatile = higher score
        out_series = -_hl_log.groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        
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
