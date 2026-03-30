import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FeatureAfdCrosswiseToStreamwise(BaseFeature):
    name = "f_afd_crosswise_to_streamwise"
    description = "AFD特徵：河彎效應中橫向渦度向流向渦度的交換"
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
        
        spread = df["SellPr"] - df["BuyPr"]
        df["_v_cross"] = spread.diff().fillna(0)
        df["_v_stream"] = np.where(df["PrFlag"] == 1, df["DealCount"].astype(float),
                           np.where(df["PrFlag"] == 0, -df["DealCount"].astype(float), 0.0))
        df["_delta_p"] = df.groupby("StockId")["DealPrice"].diff().fillna(0)
        df["_trigger"] = np.maximum(0, df["_delta_p"])
        
        df["_val"] = df["_v_cross"] * df["_v_stream"] * df["_trigger"]
        
        daily_raw = df.groupby(["StockId", "Date_int"]).agg(
            raw=("_val", "sum")
        ).reset_index()
        daily_raw = daily_raw.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
        
        out_series = daily_raw.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20)
        )
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        daily_raw[self.name] = out
        
        return daily_raw[["StockId", "Date_int", self.name]].rename(
            columns={"Date_int": "Date"}
        )
