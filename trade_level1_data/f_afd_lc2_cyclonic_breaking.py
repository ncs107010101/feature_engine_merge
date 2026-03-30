"""
F07: f_afd_lc2_cyclonic_breaking
靈感: LC2 碎波產生的強烈氣旋不對稱性
僅保留買盤壓制賣盤的動能 per bin
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_lc2_cyclonic_breaking'
BIN_SIZE = 100


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
        total_val = 0.0
        for i in range(n_bins):
            s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
            v_buy = np.sum(dc[s:e][pf[s:e] == 1])
            v_sell = np.sum(dc[s:e][pf[s:e] == 0])
            delta_v = max(0, v_buy - v_sell)
            delta_p = (dp[e - 1] - dp[s]) / (dp[s] + 1e-6)
            total_val += delta_v * delta_p

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': total_val
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=20))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
