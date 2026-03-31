import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureCrackTimeReversalAction(BaseFeature):
    name = "f_crack_time_reversal_action"
    description = (
        "時間反演路徑作用量差：S_fwd=Σ|NetFlow(t)|×|ΔP(t)|，S_rev 為時間反轉，"
        "feature=(S_fwd-S_rev)/(S_fwd+S_rev) × sign(Close-Open)。"
    )
    required_columns = ["StockId", "Date", "raw_crack_time_reversal_action"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_crack_time_reversal_action"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
