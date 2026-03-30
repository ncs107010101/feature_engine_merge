import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureMastingPredatorSatiation(BaseFeature):
    name = "f_masting_predator_satiation"
    description = (
        "Masting掠食者撐死比：爆發分鐘內主動買入量/全日被動賣出量的比率。"
        "衡量短時間突襲的力道相對於空方一整天進攻力道的強度。"
    )
    required_columns = ["StockId", "Date", "raw_burst_buy_vol", "sell_vol"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        raw = all_f["raw_burst_buy_vol"] / (all_f["sell_vol"] + 1e-8)
        out = raw.groupby(level="StockId", group_keys=False).transform(
            lambda x: zscore_rolling(x, window=42)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
