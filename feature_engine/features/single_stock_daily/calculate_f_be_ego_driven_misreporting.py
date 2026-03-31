import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FBeEgoDrivenMisreporting(BaseFeature):
    name = "f_be_ego_driven_misreporting"
    description = "Behavioral Economics: Ego-driven misreporting. When price breaks below SMA20 but retail increases margin position due to ego, trying to average down."
    required_columns = ["StockId", "Date", "收盤價", "_sma_20", "_margin_bal"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        g = df.groupby(df["StockId"])
        
        ret5 = g["收盤價"].pct_change(5)
        
        trend_down = ((df["收盤價"] < df["_sma_20"]) & (ret5 < -0.03)).astype(int)
        
        margin_diff = g["_margin_bal"].diff(1)
        
        margin_increase = (margin_diff > 0).astype(int)
        
        ego_defense = trend_down * margin_increase * margin_diff / (df["_margin_bal"] + 1e-5)
        
        out = zscore_rolling(ego_defense, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
