import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureGeoEntropyDirectedWork(BaseFeature):
    name = "f_geo_entropy_directed_work"
    description = (
        "熵驅動有向功：W_bin = F_bin × dx_bin，F=(BuyVol-SellVol)/(BuyVol+SellVol)，"
        "dx=bin 內最終-最初價格，Σ W_bin / 標準化項目。正值=訂單流方向與價格位移一致。"
    )
    required_columns = ["StockId", "Date", "raw_geo_entropy_directed_work"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_geo_entropy_directed_work"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
