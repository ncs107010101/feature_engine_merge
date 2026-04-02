import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQmMeasurementIncompatibility(BaseFeature):
    name = "f_qm_measurement_incompatibility"
    description = (
        "量子測量不相容原理：市場狀態（大戶淨買力 A_z）和交易意願（主動買比例 B_z）"
        "在相同時間窗口無法同時精確「測量」。feature = A_z × (1 - |B_z|)。"
        "需要 rolling-20日 z-score（在 calculate 中計算）。"
    )
    required_columns = [
        "StockId", "Date",
        "raw_qm_tick_active_buy_ratio",
        "raw_qm_large_broker_net_ratio"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()

        A_raw = all_f["raw_qm_large_broker_net_ratio"]
        B_raw = all_f["raw_qm_tick_active_buy_ratio"]

        # Rolling 20-day z-score (shift(1) to prevent lookahead)
        def _zscore(s):
            mu = s.shift(1).rolling(20, min_periods=5).mean()
            sd = s.shift(1).rolling(20, min_periods=5).std()
            return (s - mu) / (sd + 1e-10)

        A_z = A_raw.groupby(level="StockId").transform(_zscore)
        B_z = B_raw.groupby(level="StockId").transform(_zscore)

        out = A_z * (1.0 - B_z.abs().clip(0, 3))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
