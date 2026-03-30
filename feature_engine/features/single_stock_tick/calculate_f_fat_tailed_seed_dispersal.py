import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureFatTailedSeedDispersal(BaseFeature):
    name = "f_fat_tailed_seed_dispersal"
    description = (
        "長尾散播基因流：跨越30bps(0.3%)以上的主動買單總量/總主動買單量。"
        "衡量大額跳價買單的分散程度。"
    )
    required_columns = ["StockId", "Date", "raw_fat_tailed_seed_dispersal"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        raw = all_f["raw_fat_tailed_seed_dispersal"]
        out = raw.groupby(level="StockId", group_keys=False).transform(
            lambda x: zscore_rolling(x, window=42)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
