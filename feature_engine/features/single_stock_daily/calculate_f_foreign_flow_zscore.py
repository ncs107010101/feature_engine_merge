import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FForeignFlowZscore(BaseFeature):
    name = "f_foreign_flow_zscore"
    description = "Foreign flow zscore"
    required_columns = ["StockId", "Date", "外資買賣超張數", "_vol_ma20_orig"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        _foreign_ratio = df["外資買賣超張數"].fillna(0) / (df["_vol_ma20_orig"] + 1e-9)
        out_series = _foreign_ratio.groupby(df["StockId"]).transform(lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-9))
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)
        out = out.clip(lower=-5, upper=5)
        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
