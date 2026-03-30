"""
F14: f_afd_nlsv_zonal_shear_alignment
靈感: NLSV 產生帶狀剪切 - 大小單同向向上的瞬間協同
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_nlsv_zonal_shear_alignment'
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

        q85 = np.percentile(dc, 85)
        q50 = np.percentile(dc, 50)
        large_mask = dc >= q85
        small_mask = dc <= q50
        direction = np.where(pf == 1, 1.0, np.where(pf == 0, -1.0, 0.0))

        n_bins = n // BIN_SIZE
        total_val = 0.0
        for i in range(n_bins):
            s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
            v_large = np.sum(dc[s:e][large_mask[s:e]] * direction[s:e][large_mask[s:e]])
            v_small = np.sum(dc[s:e][small_mask[s:e]] * direction[s:e][small_mask[s:e]])
            alignment = max(0, v_large * v_small)
            total_val += alignment

        dir_sign = np.sign(dp[-1] - dp[0])
        raw_val = total_val * dir_sign

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': np.log1p(np.abs(raw_val)) * np.sign(raw_val)
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    df_res = pd.DataFrame(results).sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=20))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
