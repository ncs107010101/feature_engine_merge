import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FInverseAlleeRefugeAccumulation(BaseFeature):
    name = "f_inverse_allee_refuge_accumulation"
    description = "主力低調吸籌 = Σ[Top10_BuyQtm × P_quiet(Price)] / 總量, zscore(42). 利用5分鐘低量低波動庇護所時段識別主力隱蔽建倉。"
    required_columns = ["StockId", "Date", "raw_top10_refuge_ratio"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        raw_val = all_features["raw_top10_refuge_ratio"]

        out_series = raw_val.groupby(level="StockId").transform(
            lambda x: zscore_rolling(x, window=42, min_periods=21)
        )

        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
