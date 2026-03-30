import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import EPS

WINDOW = 20
MIN_OBS = 10
DEPTHS = [1, 2, 3, 4]


def _markov_d_logloss(b: np.ndarray, y: np.ndarray, depth: int,
                      target_class: int) -> float:
    T = len(b)
    if T <= depth + 1:
        return float("inf")

    counts = {}
    for t in range(depth, T):
        ctx = tuple(b[t - depth:t])
        yt = y[t]
        if ctx not in counts:
            counts[ctx] = [0, 0]
        counts[ctx][yt] += 1

    loss = 0.0
    n = 0
    for t in range(depth, T):
        if y[t] != target_class:
            continue
        ctx = tuple(b[t - depth:t])
        c0, c1 = counts.get(ctx, [1, 1])
        total = c0 + c1 + EPS
        p_target = (c1 + 0.5) / (total + 1) if target_class == 1 else (c0 + 0.5) / (total + 1)
        p_target = np.clip(p_target, EPS, 1 - EPS)
        loss -= np.log(p_target)
        n += 1

    return loss / n if n > 0 else float("inf")


def _compute_markov_depth_diff(win_b: np.ndarray, win_y: np.ndarray,
                               ewm_net: float) -> float:
    b = win_b[:-1]
    y = win_y[1:]
    if len(b) < MIN_OBS:
        return np.nan

    ll_up = [(d, _markov_d_logloss(b, y, d, target_class=1)) for d in DEPTHS]
    ll_dn = [(d, _markov_d_logloss(b, y, d, target_class=0)) for d in DEPTHS]

    best_up = min(ll_up, key=lambda x: x[1])[0]
    best_dn = min(ll_dn, key=lambda x: x[1])[0]

    return float((best_dn - best_up) * ewm_net)


@register_feature
class FItMarkovMemoryDepthDiff(BaseFeature):
    name = "f_it_markov_memory_depth_diff"
    description = "Markov Memory Depth Difference: (D_dn* - D_up*) * ewm_net, where D_up* is the optimal Markov depth minimizing log-loss for upward moves and D_dn* is for downward moves. Measures asymmetry in memory depth between bullish and bearish regimes."
    required_columns = [
        "StockId", "Date", "raw_big_net_ratio", "raw_p_active_up", "raw_ewm_net"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        all_features["B_seq"] = (all_features["raw_big_net_ratio"] > 0).astype(int)
        all_features["Y_seq"] = (all_features["raw_p_active_up"] > 0.5).astype(int)

        n = len(all_features)
        f18 = np.full(n, np.nan)

        B = all_features["B_seq"].values
        Y = all_features["Y_seq"].values
        ewm_arr = all_features["raw_ewm_net"].values

        for i in range(WINDOW, n):
            win_b = B[i - WINDOW:i]
            win_y = Y[i - WINDOW:i]
            ewm_today = ewm_arr[i]

            if (win_b.sum() < 2) or (win_b.sum() > WINDOW - 2):
                continue
            if (win_y.sum() < 2) or (win_y.sum() > WINDOW - 2):
                continue

            f18[i] = _compute_markov_depth_diff(win_b, win_y, ewm_today)

        out = pd.Series(f18, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
