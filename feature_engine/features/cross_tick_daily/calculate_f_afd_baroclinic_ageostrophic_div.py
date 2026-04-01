import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdBaroclinicAgeostrophicDiv(BaseFeature):
    name = "f_afd_baroclinic_ageostrophic_div"
    description = "前沿價格脫離重心的發散: max(0, close_p - vwap) × total_vol, log1p, rolling_zscore(20)"
    required_columns = ["StockId", "Date", "close_pr", "vwap", "total_vol"]
    data_combination = "cross_tick_daily"
    ZSCORE_WINDOW = 20
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        close_p = df["close_pr"].values
        vwap = df["vwap"].values
        total_vol = df["total_vol"].values
        
        raw = np.maximum(0, close_p - vwap) * total_vol
        raw = np.log1p(raw)
        raw = pd.Series(raw)
        
        out = raw.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, self.ZSCORE_WINDOW, eps=1e-10)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
