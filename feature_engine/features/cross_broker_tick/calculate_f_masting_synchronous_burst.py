import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling


@register_feature
class FMastingSynchronousBurst(BaseFeature):
    name = "f_masting_synchronous_burst"
    description = "爆發期買入HHI集中度, zscore(42). 識別日內主動買入最集中的爆發分鐘，計算Top10在此期間買入的HHI集中度。"
    required_columns = ["StockId", "Date", "raw_burst_hhi_top10"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()

        # Ensure proper time-series ordering
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        # Raw value: burst HHI
        raw_val = all_features["raw_burst_hhi_top10"]

        # Rolling z-score with window=42
        out = raw_val.groupby(level="StockId").transform(
            lambda x: zscore_rolling(x, window=42, min_periods=21)
        )

        # Clean up inf/nan
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return pd.DataFrame({
            self.name: out
        }).reset_index()
