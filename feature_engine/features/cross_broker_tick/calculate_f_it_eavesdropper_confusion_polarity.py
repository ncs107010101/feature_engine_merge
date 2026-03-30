import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import kl_divergence, EPS, N_BINS

@register_feature
class FItEavesdropperConfusionPolarity(BaseFeature):
    name = "f_it_eavesdropper_confusion_polarity"
    description = "Wiretap channel: Eavesdropper confusion intensity. KL(P_small || P_big) * ewm_net_big"
    required_columns = [
        "StockId", "Date", "raw_dist_big", "raw_dist_small", "raw_big_net_ratio"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        results = []
        for idx in all_features.index:
            dist_big = all_features.loc[idx, "raw_dist_big"]
            dist_small = all_features.loc[idx, "raw_dist_small"]
            net_big = all_features.loc[idx, "raw_big_net_ratio"]

            # Handle various data types - skip if not valid array-like
            if not isinstance(dist_big, (list, np.ndarray, pd.Series)):
                results.append(np.nan)
                continue
            if not isinstance(dist_small, (list, np.ndarray, pd.Series)):
                results.append(np.nan)
                continue
            if isinstance(dist_big, list):
                dist_big = np.array(dist_big)
            if isinstance(dist_small, list):
                dist_small = np.array(dist_small)

            # Check for valid distribution (should sum close to 1)
            try:
                big_sum = float(np.sum(dist_big))
                small_sum = float(np.sum(dist_small))
                if big_sum < 0.9 or small_sum < 0.9:
                    results.append(np.nan)
                    continue
            except (TypeError, ValueError):
                results.append(np.nan)
                continue

            kl = kl_divergence(dist_small, dist_big)
            val = kl * net_big
            results.append(val)

        out = pd.Series(results, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
