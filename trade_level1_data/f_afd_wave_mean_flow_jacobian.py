"""
F09: f_afd_wave_mean_flow_jacobian
靈感: 波與平均流交互作用導致趨勢飽和
以 5 分鐘/30分鐘 尺度的 Jacobian 行列式衡量短波動能反饋強度
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_wave_mean_flow_jacobian'
# 台股 9:00~13:30 = 32400~48600 sec, 5min=300sec
SEC_PER_BIN = 300


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 50:
            continue
        dp = df_day['DealPrice'].values
        ts = df_day['DealTimeSecond'].values

        # 按 5 分鐘切 bin
        bin_start = 32400  # 09:00:00
        bin_end = 48600    # 13:30:00
        bins_edges = np.arange(bin_start, bin_end + SEC_PER_BIN, SEC_PER_BIN)
        bin_idx = np.digitize(ts, bins_edges) - 1

        # 計算每 bin 的斜率
        slopes_5min = []
        for b in range(len(bins_edges) - 1):
            mask = bin_idx == b
            if np.sum(mask) < 3:
                slopes_5min.append(0.0)
                continue
            p_bin = dp[mask]
            x = np.arange(len(p_bin), dtype=np.float64)
            slope = np.polyfit(x, p_bin, 1)[0]
            slopes_5min.append(slope)

        slopes_5min = np.array(slopes_5min)
        if len(slopes_5min) < 7:
            results.append({'StockId': df_day['StockId'].iloc[0], 'Date': date_int, 'raw': 0.0})
            continue

        # 30分鐘斜率 (6 bins 滾動)
        slopes_30min = np.convolve(slopes_5min, np.ones(6) / 6, mode='valid')

        # Jacobian det for each pair
        min_len = min(len(slopes_5min) - 1, len(slopes_30min) - 1)
        if min_len < 2:
            results.append({'StockId': df_day['StockId'].iloc[0], 'Date': date_int, 'raw': 0.0})
            continue

        total_det = 0.0
        for i in range(min_len):
            # M = [[s5_i, s5_{i+1}], [s30_i, s30_{i+1}]]
            det_m = slopes_5min[i] * slopes_30min[i + 1] - slopes_5min[i + 1] * slopes_30min[i]
            total_det += det_m

        close_p = dp[-1]
        open_p = dp[0]
        dir_sign = np.sign(max(0, close_p - open_p))
        raw_val = total_det * dir_sign

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': raw_val
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=42))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
