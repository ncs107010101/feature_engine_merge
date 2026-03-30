"""
F20: f_afd_streamfunction_variance_filter
靈感: 流函數方差範數的信雜比濾波
大單正向推升 / 碎單正交噪聲
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_streamfunction_variance_filter'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 20:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)
        dc = df_day['DealCount'].values.astype(np.float64)

        q85 = np.percentile(dc, 85)
        q50 = np.percentile(dc, 50)

        delta_p = np.diff(dp, prepend=dp[0])
        large_mask = dc >= q85
        small_mask = dc <= q50

        # 分子: 大單正向推升
        signal = np.sum(np.maximum(0, delta_p[large_mask]) * dc[large_mask])
        # 分母: 碎單噪聲
        noise = np.sum(np.abs(delta_p[small_mask] * dc[small_mask])) + 1e-10

        raw_val = signal / noise

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
