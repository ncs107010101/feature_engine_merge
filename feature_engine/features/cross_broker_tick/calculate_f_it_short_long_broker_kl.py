import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import kl_divergence

WIN_SHORT = 5
WIN_LONG = 60
N_BINS = 5
EPS = 1e-9
NET_BINS = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]


def to_dist(arr):
    bins = pd.cut(pd.Series(arr), bins=NET_BINS, labels=False, include_lowest=True)
    d = np.bincount(bins.dropna().astype(int), minlength=N_BINS).astype(float)
    d += 1e-3
    return d / d.sum()


def compute_f21_short_long_broker_kl(net_hist):
    if len(net_hist) < WIN_LONG:
        return np.nan
    
    try:
        q5 = to_dist(net_hist[-WIN_SHORT:])
        q60 = to_dist(net_hist[-WIN_LONG:])
    except:
        return np.nan
    
    kl = kl_divergence(q5, q60)
    ewm5 = float(pd.Series(net_hist[-WIN_SHORT:]).ewm(span=5).mean().iloc[-1])
    return float(kl * ewm5)


@register_feature
class FItShortLongBrokerKl(BaseFeature):
    name = "f_it_short_long_broker_kl"
    description = "Short-Long Broker KL Divergence: KL(q_5d || q_60d) × ewm5_net. Measures distribution shift between short-term and long-term broker behavior."
    required_columns = [
        "StockId", "Date", "raw_net_ratio_hist_60d"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")
        
        f21 = np.full(len(all_features), np.nan)
        
        all_features_sorted = all_features.reset_index()
        stock_groups = all_features_sorted.groupby("StockId")
        
        for stock_id, group in stock_groups:
            indices = group.index.tolist()
            net_ratio_hist = group["raw_net_ratio_hist_60d"].values
            
            for i in range(WIN_LONG, len(indices)):
                hist = net_ratio_hist[i]
                if hist is None or (isinstance(hist, float) and np.isnan(hist)):
                    continue
                if not isinstance(hist, list) or len(hist) < WIN_LONG:
                    continue
                f21[indices[i]] = compute_f21_short_long_broker_kl(hist)
        
        out = pd.Series(f21, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
