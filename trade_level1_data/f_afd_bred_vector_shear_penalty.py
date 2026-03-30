"""
F18: f_afd_bred_vector_shear_penalty
靈感: 繁殖向量的負向剪切 - 洗盤後轉折
前段趨勢與後段擾動的幾何交叉
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_bred_vector_shear_penalty'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 20:
            continue
        dc = df_day['DealCount'].values.astype(np.float64)
        pf = df_day['PrFlag'].values
        dp = df_day['DealPrice'].values

        split = int(n * 0.6)
        direction = np.where(pf == 1, 1.0, np.where(pf == 0, -1.0, 0.0))

        # H = 前 60% ticks 的累積向量 [Σ(dc*dir), Σ(ΔP)]
        h_flow = np.sum(dc[:split] * direction[:split])
        h_price = dp[split - 1] - dp[0]
        # N = 後 40% ticks 的累積向量
        n_flow = np.sum(dc[split:] * direction[split:])
        n_price = dp[-1] - dp[split]

        # 內積 (dot)
        dot = h_flow * n_flow + h_price * n_price
        # 外積 (2D cross / det)
        cross = abs(h_flow * n_price - h_price * n_flow)

        # penalty = cross × min(0, dot) × sign(Close-Open)
        dir_sign = np.sign(dp[-1] - dp[0])
        raw_val = cross * min(0, dot) * dir_sign

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': np.log1p(np.abs(raw_val)) * np.sign(raw_val)
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    df_res = pd.DataFrame(results).sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(rolling_zscore(df_res['raw'], window=42))
    return df_res[['StockId', 'Date', FEATURE_NAME]]
