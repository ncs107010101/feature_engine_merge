import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureNetAntiferroBrokerInteraction(BaseFeature):
    name = "f_net_antiferro_broker_interaction"
    description = (
        "反鐵磁性券商交互作用：Top5買方券商集中度 vs Top5賣方券商集中度之差，"
        "× 買方淨流量佔比(買方+賣方淨流向之和的比值)。"
    )
    required_columns = ["StockId", "Date", "raw_net_antiferro_broker_interaction"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_net_antiferro_broker_interaction"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
