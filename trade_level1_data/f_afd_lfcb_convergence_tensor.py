"""
F24: f_afd_lfcb_convergence_tensor
靈感: 邊界水平輻合是垂直渦度拉伸的關鍵
買賣推進梯度的張量跡 × 正向價格
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_lfcb_convergence_tensor'
BIN_SIZE = 100


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < BIN_SIZE * 2:
            continue
        dc = df_day['DealCount'].values.astype(np.float64)
        pf = df_day['PrFlag'].values
        dp = df_day['DealPrice'].values

        n_bins = n // BIN_SIZE
        buy_totals = np.zeros(n_bins)
        sell_totals = np.zeros(n_bins)
        for i in range(n_bins):
            s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
            buy_totals[i] = np.sum(dc[s:e][pf[s:e] == 1])
            sell_totals[i] = np.sum(dc[s:e][pf[s:e] == 0])

        # 梯度 = diff
        buy_grad = np.diff(buy_totals, prepend=buy_totals[0])
        sell_grad = np.diff(sell_totals, prepend=sell_totals[0])

        # Tr(C) = buy_grad + sell_grad (輻合度)
        trace_c = buy_grad + sell_grad
        total_trace = np.sum(trace_c)

        dir_val = max(0, dp[-1] - dp[0])
        raw_val = total_trace * dir_val

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': raw_val
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    df_res = pd.DataFrame(results).sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=20))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
