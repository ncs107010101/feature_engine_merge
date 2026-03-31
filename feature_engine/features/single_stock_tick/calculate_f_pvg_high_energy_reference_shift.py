import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeaturePvgHighEnergyReferenceShift(BaseFeature):
    name = "f_pvg_high_energy_reference_shift"
    description = (
        "極端能量參考偏移：以大戶 VWAP (DealCount≥75th) 為 E0，"
        "計算全日成交量在 E0 上下的對數比值 log(vol_above/vol_below)。"
    )
    required_columns = ["StockId", "Date", "raw_pvg_high_energy_reference_shift"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_pvg_high_energy_reference_shift"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
