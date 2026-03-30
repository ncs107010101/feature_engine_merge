"""
F25: f_afd_bifurcation_tipping_point
靈感: 臨界分岔點 - 系統能量突變時的方向性觸發
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_bifurcation_tipping_point'
BIN_SIZE = 200


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < BIN_SIZE:
            continue
        dc = df_day['DealCount'].values.astype(np.float64)
        pf = df_day['PrFlag'].values
        dp = df_day['DealPrice'].values

        n_bins = n // BIN_SIZE
        if n_bins < 2:
            continue

        energies = np.zeros(n_bins)
        dp_bins = np.zeros(n_bins)
        for i in range(n_bins):
            s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
            buy_f = np.sum(dc[s:e][pf[s:e] == 1]) / BIN_SIZE
            sell_f = np.sum(dc[s:e][pf[s:e] == 0]) / BIN_SIZE
            p_s = dp[s:e]
            x = np.arange(len(p_s), dtype=np.float64)
            if len(p_s) > 5:
                sl5 = np.polyfit(x[:5], p_s[:5], 1)[0]
                sl30 = np.polyfit(x[:min(30, len(x))], p_s[:min(30, len(p_s))], 1)[0]
            else:
                sl5, sl30 = 0.0, 0.0

            energies[i] = buy_f ** 2 + sell_f ** 2 + sl5 ** 2 + sl30 ** 2
            dp_bins[i] = dp[e - 1] - dp[s]

        # 自適應閾值: mean + 2σ
        threshold = np.mean(energies) + 2 * np.std(energies)
        # 超過閾值且方向向上的 bin 之和
        exceed = np.maximum(0, energies - threshold)
        dir_sign = np.sign(np.maximum(0, dp_bins))
        raw_val = np.sum(exceed * dir_sign)

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': raw_val
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    df_res = pd.DataFrame(results).sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=42))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
