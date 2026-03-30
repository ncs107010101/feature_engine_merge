"""
F06: f_afd_moist_baroclinic_acceleration
靈感: 凝結潛熱使線性成長轉化為指數級爆發
捕捉成長率的二階加速度 (短斜率 - 長斜率) × max(0, ΔP)
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_moist_baroclinic_acceleration'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 65:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)

        # M_moist = 5-tick 短斜率
        m_moist = (dp[5:] - dp[:-5]) / 5.0
        # M_dry = 60-tick 長斜率
        m_dry_full = np.zeros(n, dtype=np.float64)
        m_dry_full[60:] = (dp[60:] - dp[:-60]) / 60.0
        m_dry = m_dry_full[5:]  # align with m_moist

        # ΔP = max(0, diff(price))
        delta_p = np.maximum(0, np.diff(dp, prepend=dp[0]))
        delta_p_aligned = delta_p[5:]

        accel = (m_moist - m_dry) * delta_p_aligned
        raw_val = np.sum(accel)

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': raw_val
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=20))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
