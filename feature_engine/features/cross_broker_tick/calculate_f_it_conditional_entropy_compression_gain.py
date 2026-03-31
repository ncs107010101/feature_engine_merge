import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import EPS

@register_feature
class FItConditionalEntropyCompressionGain(BaseFeature):
    name = "f_it_conditional_entropy_compression_gain"
    description = "Conditional entropy compression gain: Gain_up - Gain_dn (simplified daily approximation)"
    required_columns = ["StockId", "Date", "raw_big_net_ratio", "raw_p_active_up"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy().reset_index()
        
        net_ratio = df["raw_big_net_ratio"].values
        p_active_up = df["raw_p_active_up"].values
        
        B_today = (net_ratio > 0).astype(int)
        
        p_up = np.clip(p_active_up, EPS, 1 - EPS)
        adjust = 0.12 * (2 * B_today - 1)
        
        p_up_buy = np.clip(p_up + np.abs(adjust), EPS, 1 - EPS)
        p_up_sell = np.clip(p_up - np.abs(adjust), EPS, 1 - EPS)
        
        gain_up = np.log(p_up_buy / p_up)
        gain_dn = np.log((1 - p_up_sell) / (1 - p_up + EPS) + EPS)
        
        df[self.name] = gain_up - gain_dn
        
        return df[["StockId", "Date", self.name]]
