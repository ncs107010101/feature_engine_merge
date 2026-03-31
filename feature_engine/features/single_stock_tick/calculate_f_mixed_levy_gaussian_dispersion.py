import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureMixedLevyGaussianDispersion(BaseFeature):
    name = "f_mixed_levy_gaussian_dispersion"
    description = (
        "Lévy-高斯混合擴散疊加態：連續小額主動買的比例(高斯)×大額跨檔買的比例(Lévy)。"
        "衡量市場在微小波動與大幅跳躍兩種狀態間的混合程度。"
    )
    required_columns = ["StockId", "Date", "raw_mixed_levy_gaussian_dispersion"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        raw = all_f["raw_mixed_levy_gaussian_dispersion"]
        out = raw.groupby(level="StockId", group_keys=False).transform(
            lambda x: zscore_rolling(x, window=42)
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
