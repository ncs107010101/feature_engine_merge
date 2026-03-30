"""
calculate_f_it_ctw_log_likelihood_ratio.py
Group 5: CTW (Context Tree Weighting) Universal Estimator - F16

3rd Order Markov Log-Likelihood Ratio:
- Compute context = (B_{t-3}, B_{t-2}, B_{t-1}) (three binary values, 8 possible contexts)
- Calculate P(Y_t=1 | context) using Laplace smoothing
- Return log(P/(1-P)) as the feature

Reference: new_feature_code/it_g5_ctw_universal.py
"""

import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import EPS

WINDOW = 20
MIN_OBS = 10


def _compute_3markov_llr(win_b: np.ndarray, win_y: np.ndarray, context3: tuple) -> float:
    """
    Compute 3rd order Markov LLR: log(P(Y_next=1 | context) / P(Y_next=0 | context))
    context = (B_{t-3}, B_{t-2}, B_{t-1})
    """
    b = win_b[:-1]   # B_{t-1} (for predicting Y_t)
    y = win_y[1:]    # Y_t
    T = len(b)
    if T < 5:
        return np.nan

    # Build 3rd order count table: context = (B_{t-3}, B_{t-2}, B_{t-1})
    counts = {}  # context (3-tuple) -> [count_y0, count_y1]
    for t in range(3, T):
        ctx = (int(b[t - 3]), int(b[t - 2]), int(b[t - 1]))
        yt = int(y[t])
        if ctx not in counts:
            counts[ctx] = [0, 0]
        counts[ctx][yt] += 1

    # Laplace smoothing & lookup today's context
    if context3 not in counts:
        # Fallback to 1st order
        ctx1 = (context3[2],)
        n1 = sum(1 for t in range(1, T) if int(b[t - 1]) == ctx1[0])
        n1_up = sum(1 for t in range(1, T) if int(b[t - 1]) == ctx1[0] and int(y[t]) == 1)
        p_up = (n1_up + 0.5) / (n1 + 1.0)
    else:
        c0, c1 = counts[context3]
        p_up = (c1 + 0.5) / (c0 + c1 + 1.0)

    p_up = np.clip(p_up, EPS, 1 - EPS)
    return float(np.log(p_up / (1 - p_up)))


@register_feature
class FItCtwLogLikelihoodRatio(BaseFeature):
    name = "f_it_ctw_log_likelihood_ratio"
    description = "3rd Order Markov Log-Likelihood Ratio: log(P(Y=1|context)/P(Y=0|context)) where context=(B_{t-3}, B_{t-2}, B_{t-1}). Measures the predictive power of broker direction history on tick direction using context tree weighting."
    required_columns = [
        "StockId", "Date", "raw_big_net_ratio", "raw_p_active_up"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        # Create binary sequences
        all_features["B_seq"] = (all_features["raw_big_net_ratio"] > 0).astype(int)
        all_features["Y_seq"] = (all_features["raw_p_active_up"] > 0.5).astype(int)

        n = len(all_features)
        f16 = np.full(n, np.nan)

        B = all_features["B_seq"].values
        Y = all_features["Y_seq"].values

        for i in range(WINDOW, n):
            win_b = B[i - WINDOW:i]
            win_y = Y[i - WINDOW:i]

            if (win_b.sum() < 2) or (win_b.sum() > WINDOW - 2):
                continue
            if (win_y.sum() < 2) or (win_y.sum() > WINDOW - 2):
                continue

            # Need at least 3 historical points for 3rd order Markov
            if i >= 3:
                context3 = (int(B[i - 3]), int(B[i - 2]), int(B[i - 1]))
                f16[i] = _compute_3markov_llr(win_b, win_y, context3)

        out = pd.Series(f16, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
