import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import (
    binary_entropy, empirical_entropy, empirical_mi, 
    N_BINS, EPS
)

BIN_CENTERS = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])

@register_feature
class FItCompressionMismatchRedundancy(BaseFeature):
    name = "f_it_compression_mismatch_redundancy"
    description = "Normalized compression redundancy: NMI × tanh(skew_B × skew_T)"
    required_columns = [
        "StockId", "Date", "raw_dist_big", "raw_p_active_up", 
        "raw_rolling_skew_net", "raw_rolling_skew_tick"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy().reset_index()
        
        def compute_nmi(row):
            dist_big = np.array(row["raw_dist_big"])
            p_active_up = row["raw_p_active_up"]
            
            p_b = dist_big + EPS
            p_b /= p_b.sum()
            
            p_t = np.array([1 - p_active_up + EPS, p_active_up + EPS])
            p_t /= p_t.sum()
            
            p_join = np.zeros((N_BINS, 2))
            for b_idx in range(N_BINS):
                p_buy_b = np.clip(p_active_up + BIN_CENTERS[b_idx] * 0.15, 1e-3, 1 - 1e-3)
                p_join[b_idx, 1] = p_b[b_idx] * p_buy_b
                p_join[b_idx, 0] = p_b[b_idx] * (1 - p_buy_b)
            p_join = (p_join + EPS) / (p_join + EPS).sum()
            
            mi = empirical_mi(p_b, p_t, p_join)
            h_b = empirical_entropy(p_b)
            h_t = binary_entropy(np.clip(p_active_up, EPS, 1 - EPS))
            nmi = mi / (min(h_b, h_t) + EPS)
            return np.clip(nmi, 0.0, 1.0)
        
        df["nmi_val"] = df.apply(compute_nmi, axis=1)
        
        skew_b = df["raw_rolling_skew_net"].fillna(0.0)
        skew_t = df["raw_rolling_skew_tick"].fillna(0.0)
        
        df[self.name] = df["nmi_val"] * np.tanh(skew_b * skew_t)
        
        return df[["StockId", "Date", self.name]]
