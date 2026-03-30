import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdLc1FrontalGravityWave(BaseFeature):
    name = "f_afd_lc1_frontal_gravity_wave"
    description = "AFD特徵：LC1鋒面重力波"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            tick_raw = data
        
        df = tick_raw.copy()
        df["Date"] = df["Date"].astype(int)
        df = df.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
        df["Date_int"] = df["Date"]
        
        df["_direction"] = np.where(df["PrFlag"] == 1, 1.0, np.where(df["PrFlag"] == 0, -1.0, 0.0))
        df["_price_diff"] = df.groupby("StockId")["DealPrice"].diff().fillna(0)
        
        df["_stream"] = df["_direction"] * df["DealCount"]
        df["_cross"] = df["_direction"] * df["_price_diff"]
        
        def compute_gravity(group):
            stream = group["_stream"].values
            cross = group["_cross"].values
            
            if len(stream) < 10:
                return 0.0
            
            return np.sum(stream * cross)
        
        daily_raw = df.groupby(["StockId", "Date_int"]).apply(compute_gravity).reset_index(name="raw")
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
