"""
F22: f_afd_crosswise_to_streamwise
靈感: 河彎效應中橫向渦度向流向渦度的交換
掛單不平衡 × 主動買量 × 正向價格
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_crosswise_to_streamwise'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 10:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)
        dc = df_day['DealCount'].values.astype(np.float64)
        pf = df_day['PrFlag'].values
        sp = df_day['SellPr'].values
        bp = df_day['BuyPr'].values

        # V_cross = diff(spread)
        spread = sp - bp
        v_cross = np.diff(spread, prepend=spread[0])
        # V_stream = 主動買賣量
        v_stream = np.where(pf == 1, dc, np.where(pf == 0, -dc, 0.0))
        # ΔP trigger
        delta_p = np.diff(dp, prepend=dp[0])
        trigger = np.maximum(0, delta_p)

        # 特徵 = Σ(V_cross × V_stream × trigger)
        raw_val = np.sum(v_cross * v_stream * trigger)

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
