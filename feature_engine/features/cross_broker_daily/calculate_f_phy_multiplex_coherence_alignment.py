import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyMultiplexCoherenceAlignment(BaseFeature):
    name = "f_phy_multiplex_coherence_alignment"
    description = "多路相干對齊：大券商淨買率×多週期趨勢分數絕對值，衡量主力行動與多時間框架趨勢的一致性。"
    required_columns = ["StockId", "Date", "raw_phy_multiplex_coherence_alignment"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_multiplex_coherence_alignment"]]
            .copy()
            .rename(columns={"raw_phy_multiplex_coherence_alignment": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
