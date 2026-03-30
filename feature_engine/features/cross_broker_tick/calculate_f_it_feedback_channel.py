import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import EPS

WINDOW = 20
MIN_OBS = 10


def directed_info_1markov(b_seq, y_seq):
    b = np.array(b_seq, dtype=int)
    y = np.array(y_seq, dtype=int)
    T = len(b)
    if T < 5:
        return 0.0

    counts_full = np.zeros((4, 2), dtype=float)
    counts_y1 = np.zeros((2, 2), dtype=float)

    for t in range(1, T):
        ctx = b[t - 1] * 2 + y[t - 1]
        counts_full[ctx, y[t]] += 1
        counts_y1[y[t - 1], y[t]] += 1

    counts_full += 1e-3
    counts_y1 += 1e-3

    p_full = counts_full / counts_full.sum(axis=1, keepdims=True)
    p_y1 = counts_y1 / counts_y1.sum(axis=1, keepdims=True)

    di = 0.0
    for t in range(1, T):
        ctx = b[t - 1] * 2 + y[t - 1]
        yt = y[t]
        yt1 = y[t - 1]
        p_cond_full = p_full[ctx, yt]
        p_cond_base = p_y1[yt1, yt]
        di += np.log(p_cond_full / p_cond_base)

    return float(max(0.0, di / (T - 1)))


def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def empirical_mi(p_x, p_y, p_xy):
    p_x = np.array(p_x, dtype=float) + EPS
    p_y = np.array(p_y, dtype=float) + EPS
    p_xy = np.array(p_xy, dtype=float) + EPS
    p_x /= p_x.sum()
    p_y /= p_y.sum()
    p_xy /= p_xy.sum()
    hx = -np.sum(p_x * np.log(p_x))
    hy = -np.sum(p_y * np.log(p_y))
    hxy = -np.sum(p_xy * np.log(p_xy))
    return float(max(0.0, hx + hy - hxy))


@register_feature
class FItFeedbackCapacityUtilizationSkew(BaseFeature):
    name = "f_it_feedback_capacity_utilization_skew"
    description = "上行回饋利用率 = H(Y) - H(Y|Y_{t-1}=1)，特徵值 = CapUtil_up - CapUtil_dn。衡量上漲趨勢的自強化程度與下跌趨勢的自強化程度之差。"
    required_columns = [
        "StockId", "Date", "raw_feedback_capacity_skew"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        out = all_features["raw_feedback_capacity_skew"].copy()
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result


@register_feature
class FItMiConservationImbalance(BaseFeature):
    name = "f_it_mi_conservation_imbalance"
    description = "失衡量 = I(B;T) - I(B→T) - I(T→B)，特徵值 = (失衡量 / H(T)) × delta_vwap。衡量信息架構複雜性與VWAP變化的交互作用。"
    required_columns = [
        "StockId", "Date", "raw_big_net_ratio", "raw_p_active_up", "raw_delta_vwap"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        n = len(all_features)
        f24 = np.full(n, np.nan)

        B_arr = (all_features["raw_big_net_ratio"].values > 0).astype(int)
        Y_arr = (all_features["raw_p_active_up"].values > 0.5).astype(int)
        delta_vwap = all_features["raw_delta_vwap"].values

        for i in range(WINDOW, n):
            if np.isnan(delta_vwap[i]):
                continue
            win_b = B_arr[i - WINDOW:i]
            win_y = Y_arr[i - WINDOW:i]
            if (win_b.sum() < 2) or (win_b.sum() > WINDOW - 2):
                continue

            B = win_b
            Y = win_y

            counts_joint = np.zeros((2, 2), dtype=float)
            for j in range(len(B)):
                counts_joint[B[j], Y[j]] += 1
            counts_joint += 1e-3
            p_xy = counts_joint / counts_joint.sum()
            p_x = p_xy.sum(axis=1)
            p_y = p_xy.sum(axis=0)
            MI = empirical_mi(p_x, p_y, p_xy)

            DI_bt = directed_info_1markov(B, Y)
            DI_tb = directed_info_1markov(Y, B)

            imbalance = MI - DI_bt - DI_tb

            H_T = binary_entropy(np.clip(Y.mean(), EPS, 1 - EPS))
            norm_imbalance = imbalance / (H_T + EPS)

            f24[i] = float(norm_imbalance * delta_vwap[i])

        out = pd.Series(f24, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result


@register_feature
class FItBroadcastChannelEntropyGap(BaseFeature):
    name = "f_it_broadcast_channel_entropy_gap"
    description = "H_small - H_large（散戶混亂、主力方向明確 → 正值）× net_ratio_big。衡量散戶與主力交易行為的熵差異。"
    required_columns = [
        "StockId", "Date", "raw_p_large_buy", "raw_p_small_buy", "raw_big_net_ratio"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        p_large_buy = all_features["raw_p_large_buy"].values
        p_small_buy = all_features["raw_p_small_buy"].values
        net_ratio_big = all_features["raw_big_net_ratio"].values

        H_large = np.array([binary_entropy(np.clip(p, EPS, 1 - EPS)) for p in p_large_buy])
        H_small = np.array([binary_entropy(np.clip(p, EPS, 1 - EPS)) for p in p_small_buy])

        entropy_gap = H_small - H_large
        out = pd.Series(entropy_gap * net_ratio_big, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
