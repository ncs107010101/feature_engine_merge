"""
F19: f_afd_sv_optimization_time_divergence
靈感: 最佳擾動對最佳化時間的高度敏感性
短期推動力遠超長期時觸發
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_sv_optimization_time_divergence'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 65:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)
        dc = df_day['DealCount'].values.astype(np.float64)

        delta_p = np.diff(dp, prepend=dp[0])

        # EWM spans: short=5, long=60
        # Vectorized EWM approximation
        alpha_s = 2.0 / (5 + 1)
        alpha_l = 2.0 / (60 + 1)
        v_short = np.zeros(n, dtype=np.float64)
        v_long = np.zeros(n, dtype=np.float64)
        v_short[0] = delta_p[0]
        v_long[0] = delta_p[0]
        for i in range(1, n):
            v_short[i] = alpha_s * delta_p[i] + (1 - alpha_s) * v_short[i - 1]
            v_long[i] = alpha_l * delta_p[i] + (1 - alpha_l) * v_long[i - 1]

        divergence = v_short - v_long
        # max(0, divergence × DealCount)
        weighted = np.maximum(0, divergence * dc)
        raw_val = np.sum(weighted)

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': np.log1p(raw_val)
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    df_res = pd.DataFrame(results).sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=20))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
