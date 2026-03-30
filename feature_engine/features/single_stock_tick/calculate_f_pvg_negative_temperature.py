import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeaturePvgNegativeTemperature(BaseFeature):
    name = "f_pvg_negative_temperature"
    description = (
        "負溫度微結構反轉：corr(|Price-Open|, NetFlow) × 60日受限度(vol/max_vol_60)。"
        "正常狀態該相關為負；高量縮期間若相關反轉則代表負溫度信號。"
    )
    required_columns = ["StockId", "Date", "raw_pvg_negative_temperature_corr", "total_vol"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()

        temp_corr = all_f["raw_pvg_negative_temperature_corr"]

        # 60-day confinement: daily_vol / rolling-max-60-of-past-vol (shift(1) to avoid lookahead)
        daily_vol = all_f["total_vol"]
        max_vol_60 = (daily_vol.groupby(level="StockId")
                      .transform(lambda x: x.shift(1).rolling(60, min_periods=5).max()))
        confinement = (daily_vol / (max_vol_60 + 1e-10)).clip(0, 3)

        out = temp_corr * confinement
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
