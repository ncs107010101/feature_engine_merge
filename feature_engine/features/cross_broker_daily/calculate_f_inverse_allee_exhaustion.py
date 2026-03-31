import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FInverseAlleeExhaustion(BaseFeature):
    name = "f_inverse_allee_exhaustion"
    description = "逆向阿利效應枯竭 - 股價創20日新低時，小型逆勢買家與大型順勢賣家的比率趨勢"
    required_columns = ["StockId", "Date", "raw_is_new_low_20d", "raw_small_buyers_cnt", "raw_large_sellers_cnt"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        ratio = df["raw_small_buyers_cnt"] / (df["raw_large_sellers_cnt"] + 1.0)
        
        ratio_filled = np.where(
            df["raw_is_new_low_20d"] == 1.0,
            ratio,
            np.nan
        )
        ratio_series = pd.Series(ratio_filled, index=df.index)
        ratio_series = ratio_series.groupby(df["StockId"]).transform(lambda x: x.ffill().fillna(0))
        
        def calc_slope(y):
            if len(y) < 2:
                return 0.0
            x = np.arange(len(y))
            if np.all(y == y.iloc[0]):
                return 0.0
            return np.polyfit(x, y, 1)[0]
        
        slope_10 = ratio_series.groupby(df["StockId"]).transform(
            lambda x: x.rolling(window=10, min_periods=2).apply(calc_slope, raw=False)
        ).fillna(0)
        
        out_series = slope_10.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, 42)
        )
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
