import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FDilutionEffectDefense(BaseFeature):
    name = "f_dilution_effect_defense"
    description = "生產者防禦稀釋效應 - 砸盤日時淨買入券商數除以大戶淨賣出量的比例"
    required_columns = ["StockId", "Date", "raw_is_dump_day", "raw_buyer_count", "raw_top10_net_sell_qtm"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        eps = 1e-8
        
        raw = np.where(
            df["raw_is_dump_day"] == 1.0,
            df["raw_buyer_count"] / (df["raw_top10_net_sell_qtm"] + eps),
            0.0
        )
        
        raw_series = pd.Series(raw, index=df.index)
        raw_ewm = raw_series.groupby(df["StockId"]).transform(
            lambda x: x.ewm(span=10, min_periods=1).mean()
        )
        
        out_series = raw_series.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, 42)
        )
        
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
