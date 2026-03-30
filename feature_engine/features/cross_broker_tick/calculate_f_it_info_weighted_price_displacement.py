import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

EPS = 1e-9


def binary_h(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def compute_f20_info_weighted_price_displacement(segments):
    if not segments or len(segments) < 8:
        return np.nan
    
    result = 0.0
    for s in range(1, 8):
        h_prev = segments[s - 1].get("h", np.nan) if isinstance(segments[s - 1], dict) else np.nan
        h_curr = segments[s].get("h", np.nan) if isinstance(segments[s], dict) else np.nan
        v_prev = segments[s - 1].get("vwap", np.nan) if isinstance(segments[s - 1], dict) else np.nan
        v_curr = segments[s].get("vwap", np.nan) if isinstance(segments[s], dict) else np.nan
        
        if any(np.isnan([h_prev, h_curr, v_prev, v_curr])):
            continue
        if v_prev < EPS:
            continue
        dh = h_prev - h_curr
        dp = v_curr / v_prev - 1.0
        if dh > 0:
            result += dh * dp
    
    return float(result)


@register_feature
class FItInfoWeightedPriceDisplacement(BaseFeature):
    name = "f_it_info_weighted_price_displacement"
    description = "Information Weighted Price Displacement: Sum of entropy drop × price change across 30-min segments. Measures information flow intensity weighted by price movement."
    required_columns = [
        "StockId", "Date", "raw_tick_segments"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        
        f20 = all_features["raw_tick_segments"].apply(
            lambda segs: compute_f20_info_weighted_price_displacement(segs) if isinstance(segs, list) else np.nan
        )
        
        f20 = f20.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: f20.values
        }, index=all_features.index).reset_index()
        
        return final_result
