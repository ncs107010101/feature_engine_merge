import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ts_rank_center


@register_feature
class FRetailHeatComposite(BaseFeature):
    name = "f_retail_heat_composite"
    description = "Retail heat composite (time-series normalized)"
    required_columns = ["StockId", "Date", "融資餘額(千股)", "_vol_ma20", "_tr_z", "_dt_z"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        _mb = g["融資餘額(千股)"].ffill()
        _mc_raw = _mb.groupby(df["StockId"]).diff(5) / (df["_vol_ma20"] + 1e-9)
        _mc_z = _mc_raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=15, min_periods=5))
        
        # Use time-series rank instead of cross-sectional rank
        r_tr = df["_tr_z"].groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        r_dt = df["_dt_z"].groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        r_mc = _mc_z.groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        
        # Negate: higher retail heat = more negative score (anti-retail)
        out_series = -(r_tr * 0.40 + r_dt * 0.35 + r_mc * 0.25)
        
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
