import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureITLargeTradeIATAsymmetry(BaseFeature):
    name = "f_it_large_trade_iat_asymmetry"
    description = (
        "大額成交（>=90th）IAT（Inter-Arrival Time）加速度不對稱。"
        "IAT_accel = 前半段IAT / 後半段IAT，正值=賣方相對更加速。"
    )
    required_columns = ["StockId", "Date", "raw_it_large_trade_iat_asymmetry"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_it_large_trade_iat_asymmetry"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
