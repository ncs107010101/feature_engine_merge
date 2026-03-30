"""
F08: f_afd_lc1_frontal_gravity_wave
靈感: LC1 碎波中冷鋒激發的反向慣性重力波
外積衡量掛單壓力與價格位移的正交程度
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_lc1_frontal_gravity_wave'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 10:
            continue
        dp = df_day['DealPrice'].values
        sp = df_day['SellPr'].values
        bp = df_day['BuyPr'].values

        spread = sp - bp
        grad_lob = np.diff(spread, prepend=spread[0])
        delta_p = np.diff(dp, prepend=dp[0])

        # 2D cross product approximation using consecutive pairs
        # cross = |grad_lob_i × delta_p_{i+1} - delta_p_i × grad_lob_{i+1}|
        cross_vals = np.abs(
            grad_lob[:-1] * delta_p[1:] - delta_p[:-1] * grad_lob[1:]
        )
        sign_dp = np.sign(delta_p[:-1])
        weighted_cross = cross_vals * sign_dp

        raw_val = np.sum(weighted_cross)

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
