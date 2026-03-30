import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeaturePolSpectralCentroidShift(BaseFeature):
    name = "f_pol_spectral_centroid_shift"
    description = (
        "動量場頻譜重心偏移：對 20-tick bins 動量序列做 FFT，計算頻率加權重心，"
        "× sign(close-open)。高頻重心=短週期震盪；正值=上漲方向的高頻動能。"
        "額外加 rolling-20日 z-score 標準化。"
    )
    required_columns = ["StockId", "Date", "raw_pol_spectral_centroid_shift"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        raw = all_f["raw_pol_spectral_centroid_shift"]

        # Rolling 20-day z-score (use shift(1) to avoid lookahead)
        roll_mean = raw.groupby(level="StockId").transform(
            lambda x: x.shift(1).rolling(20, min_periods=5).mean())
        roll_std = raw.groupby(level="StockId").transform(
            lambda x: x.shift(1).rolling(20, min_periods=5).std())
        out = (raw - roll_mean) / (roll_std + 1e-10)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
