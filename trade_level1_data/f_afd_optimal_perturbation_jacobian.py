"""
F13: f_afd_optimal_perturbation_jacobian
靈感: 最佳擾動在有限時間內產生最大能量增幅
2×2 斜率矩陣 (價格+成交量) 的行列式
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_optimal_perturbation_jacobian'
BIN_SIZE = 50


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < BIN_SIZE * 2:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)
        dc = df_day['DealCount'].values.astype(np.float64)

        n_bins = n // BIN_SIZE
        total_det = 0.0
        for i in range(n_bins):
            s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
            p_s, p_e = dp[s], dp[e - 1]
            v_s, v_e = np.sum(dc[s:s + 5]), np.sum(dc[e - 5:e])
            v_long_s, v_long_e = np.sum(dc[s:s + 30]), np.sum(dc[max(s, e - 30):e])

            # Slopes
            slope_5p = (dp[min(s + 5, e - 1)] - dp[s]) / 5.0
            slope_30p = (dp[min(s + 30, e - 1)] - dp[s]) / 30.0
            slope_5v = (v_e - v_s) / (5.0 + 1e-10)
            slope_30v = (v_long_e - v_long_s) / (30.0 + 1e-10)

            # det = slope_5p * slope_30v - slope_30p * slope_5v
            det_s = slope_5p * slope_30v - slope_30p * slope_5v
            total_det += det_s

        dir_val = max(0, dp[-1] - dp[0])
        raw_val = total_det * dir_val

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
