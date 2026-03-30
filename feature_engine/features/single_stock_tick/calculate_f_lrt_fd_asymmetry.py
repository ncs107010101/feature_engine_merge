import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureLrtFdAsymmetry(BaseFeature):
    name = "f_lrt_fd_asymmetry"
    description = (
        "漲落耗散非對稱比：分別計算上漲/下跌 tick 的 dp_per_vol 分佈響應效率"
        "(|mean|/std)，取兩者之差的標準化值。正值=上漲效率高→看漲。"
    )
    required_columns = ["StockId", "Date", "raw_lrt_fd_asymmetry"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_lrt_fd_asymmetry"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
