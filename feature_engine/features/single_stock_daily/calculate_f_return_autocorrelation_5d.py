import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FPriceVolumeCorr20d(BaseFeature):
    name = "f_price_volume_corr_20d"
    description = "Price-volume rolling correlation (20-day)"
    required_columns = ["StockId", "Date", "成交量(千股)", "_ret_1d"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        df_copy = df.copy()
        df_copy["_vol_pct_change"] = g["成交量(千股)"].pct_change()
        _pv_corr = pd.Series(np.nan, index=df.index)
        for sid, sidx in g.groups.items():
            ret_s = df_copy.loc[sidx, "_ret_1d"]
            vol_s = df_copy.loc[sidx, "_vol_pct_change"]
            _pv_corr.loc[sidx] = ret_s.rolling(20, min_periods=10).corr(vol_s)
        out_series = _pv_corr
        
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
