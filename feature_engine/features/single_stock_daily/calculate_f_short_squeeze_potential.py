import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FShortSqueezePotential(BaseFeature):
    name = "f_short_squeeze_potential"
    description = "Short squeeze potential"
    required_columns = ["StockId", "Date", "融券餘額(千股)", "_vol_ma20"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        _short_bal = g["融券餘額(千股)"].ffill().fillna(0)
        _cover_days = _short_bal / (df["_vol_ma20"] + 1e-8)
        _log_cover = np.log1p(_cover_days)
        _ssp = pd.Series(np.nan, index=df.index)
        for sid, sidx in g.groups.items():
            s = _log_cover.loc[sidx]
            _ssp.loc[sidx] = s.rolling(20, min_periods=5).apply(lambda w: pd.Series(w).rank(pct=True).iloc[-1])
        out_series = _ssp
        
        # Post-processing
        out = out_series
        out = out.replace([np.inf, -np.inf], np.nan)
        # Note: we group ffill by StockId so it does not bleed across stocks
        out = out.groupby(df["StockId"]).ffill().fillna(0.0)

        
        result = pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
        return result
