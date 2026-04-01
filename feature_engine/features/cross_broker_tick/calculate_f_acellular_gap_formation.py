import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FAcellularGapFormation(BaseFeature):
    name = "f_acellular_gap_formation"
    description = "Z-score of cumsum(spread_on_downtick × top10_net_sell_ratio). 空方倒貨時買方撤退形成的內盤委賣價差擴大。"
    required_columns = ["StockId", "Date", "raw_top10_net_sell_ratio", "raw_mean_spread_downtick"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        # Compute raw feature: spread × sell ratio
        raw_val = all_features["raw_top10_net_sell_ratio"] * all_features["raw_mean_spread_downtick"]

        # Cumulative sum per stock
        cumsum_val = raw_val.groupby(level="StockId").cumsum()

        # Rolling z-score with window=42
        out = cumsum_val.groupby(level="StockId").transform(
            lambda x: zscore_rolling(x, window=42, min_periods=21)
        )

        # Clean up inf/nan
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return pd.DataFrame({
            self.name: out
        }).reset_index()
