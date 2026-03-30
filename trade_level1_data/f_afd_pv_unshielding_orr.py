"""
F15: f_afd_pv_unshielding_orr
靈感: 奧爾效應中相位速度差異導致位渦去屏蔽
掛單結構與市價成交的正交度
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_pv_unshielding_orr'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 10:
            continue
        dp = df_day['DealPrice'].values
        bp = df_day['BuyPr'].values
        sp = df_day['SellPr'].values

        mid_price = (bp + sp) / 2.0
        c_limit = np.diff(mid_price, prepend=mid_price[0])
        c_market = np.diff(dp, prepend=dp[0])

        # 2D cross: |c_limit_i * c_market_{i+1} - c_market_i * c_limit_{i+1}|
        cross = np.abs(c_limit[:-1] * c_market[1:] - c_market[:-1] * c_limit[1:])
        delta_p_trigger = np.maximum(0, c_market[:-1])
        weighted = cross * delta_p_trigger

        raw_val = np.sum(weighted)

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': raw_val
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    df_res = pd.DataFrame(results).sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=20))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
