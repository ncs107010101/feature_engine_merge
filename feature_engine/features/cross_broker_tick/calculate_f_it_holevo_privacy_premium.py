import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import EPS

@register_feature
class FItHolevoPrivacyPremium(BaseFeature):
    name = "f_it_holevo_privacy_premium"
    description = "Wiretap channel: Holevo information premium difference. χ_up - χ_dn, normalized to [-1, +1]"
    required_columns = [
        "StockId", "Date", "raw_broker_weights_big", "raw_broker_nets_big", "raw_p_active_up"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        def broker_state_dist(broker_net, p_bias):
            p_buy_k = np.clip(p_bias + broker_net * 0.2, 1e-3, 1 - 1e-3)
            return np.array([1 - p_buy_k, p_buy_k])

        def holevo_chi(weights, nets, p_env):
            dists = [broker_state_dist(n, p_env) for n in nets]
            mixed = sum(w * d for w, d in zip(weights, dists))
            mixed = np.clip(mixed, 1e-9, None)
            mixed /= mixed.sum()
            H_mixed = -np.sum(mixed * np.log2(mixed + 1e-12))
            avg_H = sum(w * (-np.sum(np.clip(d, 1e-9, None) *
                                      np.log2(np.clip(d, 1e-9, None))))
                        for w, d in zip(weights, dists))
            return max(0.0, H_mixed - avg_H)

        results = []
        for idx in all_features.index:
            weights_data = all_features.loc[idx, "raw_broker_weights_big"]
            nets_data = all_features.loc[idx, "raw_broker_nets_big"]
            p_active_up = all_features.loc[idx, "raw_p_active_up"]

            if isinstance(weights_data, list):
                weights = np.array(weights_data)
            else:
                weights = np.array([1.0])
            if isinstance(nets_data, list):
                nets = np.array(nets_data)
            else:
                nets = np.array([0.0])

            K = len(weights)
            if K < 2:
                results.append(np.nan)
                continue

            weights = weights / weights.sum()

            chi_up = holevo_chi(weights, nets, p_env=p_active_up)
            chi_dn = holevo_chi(weights, nets, p_env=1 - p_active_up)

            max_chi = np.log2(max(K, 2))
            val = (chi_up - chi_dn) / (max_chi + EPS)
            results.append(val)

        out = pd.Series(results, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
