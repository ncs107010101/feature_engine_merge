import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureITPostEventDirectionalImpulse(BaseFeature):
    name = "f_it_post_event_directional_impulse"
    description = (
        "大額成交事件（>= 90th DealCount）後10筆Tick的方向衝擊。"
        "特徵值 = μ_buy_event - μ_sell_event，衡量大單後市場反應的方向性。"
    )
    required_columns = ["StockId", "Date", "raw_it_post_event_directional_impulse"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_it_post_event_directional_impulse"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
