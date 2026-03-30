import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import rolling_zscore


@register_feature
class FeatureDensityDependentDispersalDamping(BaseFeature):
    name = "f_density_dependent_dispersal_damping"
    description = "密度依賴性擴散阻尼。日內最大成交價位的成交佔比與價格脫離該區間幅度的標準化差。"
    required_columns = ["StockId", "Date", "raw_mode_share", "raw_mode_price", "收盤價"]
    data_combination = "cross_tick_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        EscapeDist = (df["收盤價"] - df["raw_mode_price"]).abs() / (df["raw_mode_price"] + 1e-8)
        
        z_mode_share = df["raw_mode_share"].groupby(df["StockId"]).transform(lambda x: rolling_zscore(x, 42))
        z_escape = EscapeDist.fillna(0).groupby(df["StockId"]).transform(lambda x: rolling_zscore(x, 42))
        
        raw = z_mode_share - z_escape
        raw = raw.fillna(0)
        out = raw.groupby(df["StockId"]).transform(lambda x: rolling_zscore(x, 42))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
