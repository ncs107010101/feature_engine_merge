import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeaturePredatorMateLimitation(BaseFeature):
    name = "f_predator_mate_limitation"
    description = (
        "掠食者配偶限制：空方大戶在急跌段(1分鐘return<-0.3%)找不到對手盤(買方)，"
        "做空效率大降。plunge_sell_hhi/(plunge_buy_depth+ε)。"
    )
    required_columns = ["StockId", "Date", "raw_plunge_sell_hhi", "raw_plunge_buy_depth"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        raw = all_f["raw_plunge_sell_hhi"] / (all_f["raw_plunge_buy_depth"] + 1e-8)
        out = raw.groupby(level="StockId", group_keys=False).transform(
            lambda x: zscore_rolling(x, window=42)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
