import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import rolling_zscore


@register_feature
class FeatureLimitCycleExhaustionTrap(BaseFeature):
    name = "f_limit_cycle_exhaustion_trap"
    description = "動力學極限環耗散枯竭。(VWAP穿越次數 * 總成交量) / 價格位移幅度，取對數後Z-score。"
    required_columns = ["StockId", "Date", "raw_cross_count", "total_vol", "收盤價", "開盤價"]
    data_combination = "cross_tick_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        Displacement = (df["收盤價"] - df["開盤價"]).abs() / (df["開盤價"] + 1e-8)
        
        raw = (df["raw_cross_count"] * df["total_vol"]) / (Displacement + 1e-8)
        log_raw = np.log1p(raw)
        
        out = log_raw.groupby(df["StockId"]).transform(lambda x: rolling_zscore(x, 42))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
