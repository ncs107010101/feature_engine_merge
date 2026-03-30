import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FValueComposite(BaseFeature):
    name = "f_value_composite"
    description = "Value composite (time-series normalized)"
    required_columns = ["StockId", "Date", "本益比(TEJ)", "股價淨值比(TEJ)", "股價營收比(TEJ)", "現金股利率(TEJ)"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        # Use time-series z-score instead of cross-sectional rank
        # Lower PE/PBR/PSR = more value → negate z-score
        z_pe = -g["本益比(TEJ)"].transform(lambda x: zscore_rolling(x, window=120, min_periods=20))
        z_pbr = -g["股價淨值比(TEJ)"].transform(lambda x: zscore_rolling(x, window=120, min_periods=20))
        z_psr = -g["股價營收比(TEJ)"].transform(lambda x: zscore_rolling(x, window=120, min_periods=20))
        # Higher dividend yield = more value → keep sign
        z_dy = g["現金股利率(TEJ)"].transform(lambda x: zscore_rolling(x, window=120, min_periods=20))
        
        out_series = z_pe * 0.30 + z_pbr * 0.25 + z_psr * 0.25 + z_dy * 0.20
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=-5, upper=5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
