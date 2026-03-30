"""
F11: f_afd_vortex_dipole_ejection
靈感: 渦旋偶極子彈射 - 早盤/尾盤流動張量的正交性
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_vortex_dipole_ejection'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 20:
            continue
        dc = df_day['DealCount'].values.astype(np.float64)
        pf = df_day['PrFlag'].values
        dp = df_day['DealPrice'].values

        mid = n // 2
        # AM portion
        am_buy = np.sum(dc[:mid][pf[:mid] == 1])
        am_sell = np.sum(dc[:mid][pf[:mid] == 0])
        # PM portion
        pm_buy = np.sum(dc[mid:][pf[mid:] == 1])
        pm_sell = np.sum(dc[mid:][pf[mid:] == 0])

        # Tr(F_AM ⊗ F_PM) = am_buy*pm_buy + am_sell*pm_sell
        trace_val = am_buy * pm_buy + am_sell * pm_sell
        dir_val = max(0, dp[-1] - dp[0])
        raw_val = trace_val * dir_val

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': np.log1p(raw_val)  # log compress
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    df_res = pd.DataFrame(results).sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=42))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
