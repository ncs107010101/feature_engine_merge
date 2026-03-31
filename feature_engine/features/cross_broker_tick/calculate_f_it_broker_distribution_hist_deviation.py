import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import kl_divergence

WINDOW = 20
N_BINS = 5
EPS = 1e-9
NET_BINS = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]


def compute_f22_broker_dist_hist_deviation(net_hist, net_today_val):
    if len(net_hist) < WINDOW:
        return np.nan
    
    hist_window = net_hist[-WINDOW:]
    mu_20 = np.mean(hist_window)
    sigma_20 = np.std(hist_window) + EPS
    z = (net_today_val - mu_20) / sigma_20
    
    bin_centers = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    today_dist = np.exp(-0.5 * ((net_today_val - bin_centers) / 0.3) ** 2) + 1e-3
    today_dist /= today_dist.sum()
    
    hist_net = hist_window
    bins_h = pd.cut(pd.Series(hist_net), bins=NET_BINS, labels=False, include_lowest=True)
    hist_dist = np.bincount(bins_h.dropna().astype(int), minlength=N_BINS).astype(float)
    hist_dist += 1e-3
    hist_dist /= hist_dist.sum()
    
    kl = kl_divergence(today_dist, hist_dist)
    return float(kl * np.tanh(z))


@register_feature
class FItBrokerDistributionHistDeviation(BaseFeature):
    name = "f_it_broker_distribution_hist_deviation"
    description = "Broker Distribution Historical Deviation: KL(today_dist || 20d_median_dist) × tanh(z). Measures anomaly in today's broker distribution compared to historical patterns."
    required_columns = [
        "StockId", "Date", "raw_net_ratio_hist_20d", "raw_big_net_ratio"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")
        
        f22 = np.full(len(all_features), np.nan)
        
        all_features_sorted = all_features.reset_index()
        stock_groups = all_features_sorted.groupby("StockId")
        
        for stock_id, group in stock_groups:
            indices = group.index.tolist()
            net_hist = group["raw_net_ratio_hist_20d"].values
            net_today = group["raw_big_net_ratio"].values
            
            for i in range(WINDOW, len(indices)):
                hist = net_hist[i]
                if hist is None or (isinstance(hist, float) and np.isnan(hist)):
                    continue
                if not isinstance(hist, list) or len(hist) < WINDOW:
                    continue
                net_val = net_today[i]
                if np.isnan(net_val):
                    continue
                f22[indices[i]] = compute_f22_broker_dist_hist_deviation(hist, net_val)
        
        out = pd.Series(f22, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()
        
        return final_result
