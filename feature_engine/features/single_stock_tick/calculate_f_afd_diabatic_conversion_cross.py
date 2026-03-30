import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

EPS = 1e-10


@register_feature
class FeatureAfdDiabaticConversionCross(BaseFeature):
    name = "f_afd_diabatic_conversion_cross"
    description = "AFD特徵：DRV非絕熱加熱轉換率"
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
        
        df["_V"] = np.where(df["PrFlag"] == 1, df["DealCount"].astype(float),
                    np.where(df["PrFlag"] == 0, -df["DealCount"].astype(float), 0.0))
        df["_delta_p"] = df.groupby("StockId")["DealPrice"].diff().fillna(0)
        df["_G_E"] = df["_V"] * np.maximum(0, df["_delta_p"])
        df["_C_A"] = df["_V"] * np.abs(df["_delta_p"])
        df["_diff"] = df["_G_E"] - df["_C_A"]
        
        daily_raw = df.groupby(["StockId", "Date_int"])["_diff"].sum().reset_index()
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["_diff"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
