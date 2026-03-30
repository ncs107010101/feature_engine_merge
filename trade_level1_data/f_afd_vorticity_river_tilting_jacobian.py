"""
F21: f_afd_vorticity_river_tilting_jacobian
靈感: 渦度河流中橫向渦度被向上傾斜轉化
Jacobian det 衡量量價流形拉伸
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_vorticity_river_tilting_jacobian'
BIN_SIZE = 50


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < BIN_SIZE * 2:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)
        dc = df_day['DealCount'].values.astype(np.float64)
        ts = df_day['DealTimeSecond'].values.astype(np.float64)

        n_bins = n // BIN_SIZE
        total_det = 0.0
        for i in range(n_bins):
            s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
            dv = np.sum(dc[s:e]) - np.sum(dc[max(0, s - BIN_SIZE):s]) if i > 0 else np.sum(dc[s:e])
            d_price = dp[e - 1] - dp[s]
            dt = ts[e - 1] - ts[s] + 1e-6

            # det(J) = (ΔV/Δt) - (ΔV/ΔP) × (ΔP/Δt)
            dv_dt = dv / dt
            dp_dt = d_price / dt
            dv_dp = dv / (d_price + 1e-8)
            det_j = dv_dt - dv_dp * dp_dt
            total_det += det_j

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
