import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitFlowEmbezzlementProxy(BaseFeature):
    name = "f_qit_flow_embezzlement_proxy"
    description = "訂單流盜用指標。小額主動買單創造的向上Tick數量佔比。"
    
    required_columns = ["StockId", "Date", "raw_qit_flow_embezzlement_proxy"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_flow_embezzlement_proxy"]].copy()
        df.rename(columns={"raw_qit_flow_embezzlement_proxy": self.name}, inplace=True)
        return df
