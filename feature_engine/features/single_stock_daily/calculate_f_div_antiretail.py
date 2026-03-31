import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling, ts_rank_center


@register_feature
class FDivAntiretail(BaseFeature):
    name = "f_div_antiretail"
    description = "Div antiretail (time-series normalized)"
    required_columns = ["StockId", "Date", "投信買賣超張數", "外資買賣超張數", "_vol_ma10", "_tr_z", "_dt_z"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        g = df.groupby("StockId", group_keys=False)
        
        trust = df["投信買賣超張數"].fillna(0)
        foreign = df["外資買賣超張數"].fillna(0)
        divergence = (np.sign(trust) != np.sign(foreign)).astype(float)
        magnitude = (trust.abs() + foreign.abs()) / (df["_vol_ma10"] + 1e-9)
        _div_w = divergence * magnitude
        _div_ewm = _div_w.groupby(df["StockId"]).transform(lambda x: x.ewm(span=5, min_periods=3).mean())
        
        # Use time-series rank instead of cross-sectional rank
        r_div = g[_div_ewm.name if hasattr(_div_ewm, 'name') else 0].transform(lambda x: x) if False else _div_ewm.groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        r_tr = df["_tr_z"].groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        r_dt = df["_dt_z"].groupby(df["StockId"]).transform(lambda x: ts_rank_center(x, window=120))
        
        # r_div higher = more divergence (good for anti-retail)
        # r_tr, r_dt higher = more retail heat → negate
        out_series = r_div * 0.40 + (-r_tr) * 0.30 + (-r_dt) * 0.30
        
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
