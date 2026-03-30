import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import empirical_mi, binary_entropy, kl_divergence, EPS, N_BINS

SMOOTH_EPS = 1e-3

@register_feature
class FItSecrecyCapacityRate(BaseFeature):
    name = "f_it_secrecy_capacity_rate"
    description = "Wiretap channel: Standardized secrecy capacity rate. Cs = max(0, I(X;Y_large) - I(X;Y_small)) / H(T) * net_ratio_big"
    required_columns = [
        "StockId", "Date", "raw_dist_big", "raw_p_large_buy", "raw_p_small_buy",
        "raw_p_active_up", "raw_big_net_ratio"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        bin_centers = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])

        def compute_mi_x_y(p_x_dist, p_y_dist):
            p_buy_overall = p_y_dist[1] / p_y_dist.sum()
            p_xy = np.zeros((N_BINS, 2))
            for b_idx in range(N_BINS):
                p_buy_b = np.clip(p_buy_overall + bin_centers[b_idx] * 0.15, SMOOTH_EPS, 1 - SMOOTH_EPS)
                p_xy[b_idx, 1] = p_x_dist[b_idx] * p_buy_b
                p_xy[b_idx, 0] = p_x_dist[b_idx] * (1 - p_buy_b)
            p_xy = (p_xy + 1e-9) / (p_xy + 1e-9).sum()
            return empirical_mi(p_x_dist, p_y_dist / p_y_dist.sum(), p_xy)

        results = []
        for idx in all_features.index:
            p_x = all_features.loc[idx, "raw_dist_big"]
            p_large = all_features.loc[idx, "raw_p_large_buy"]
            p_small = all_features.loc[idx, "raw_p_small_buy"]
            p_active_up = all_features.loc[idx, "raw_p_active_up"]
            net_ratio_big = all_features.loc[idx, "raw_big_net_ratio"]

            if isinstance(p_x, list):
                p_x = np.array(p_x)

            p_y_large = np.array([1 - p_large + SMOOTH_EPS, p_large + SMOOTH_EPS])
            p_y_small = np.array([1 - p_small + SMOOTH_EPS, p_small + SMOOTH_EPS])

            I_x_large = compute_mi_x_y(p_x, p_y_large)
            I_x_small = compute_mi_x_y(p_x, p_y_small)

            Cs = I_x_large - I_x_small

            H_T = binary_entropy(np.clip(p_active_up, EPS, 1 - EPS))
            if H_T < EPS:
                val = np.nan
            else:
                val = (Cs / H_T) * net_ratio_big

            results.append(val)

        out = pd.Series(results, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
