"""
F23: f_afd_vortex_line_slippage
靈感: 氣塊穿過冷池邊界的渦度滑移
破底翻三位一體: 早盤砸盤 → 中盤吸籌 → 尾盤拉升
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_vortex_line_slippage'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 30:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)
        dc = df_day['DealCount'].values.astype(np.float64)
        pf = df_day['PrFlag'].values

        third = n // 3
        # 早盤(前1/3): 砸盤動能 (負值)
        early_ret = dp[third - 1] - dp[0]
        early_sell = np.sum(dc[:third][pf[:third] == 0])
        D = min(0, early_ret) * early_sell  # 負值

        # 中盤(中1/3): 大單吸籌量
        mid_buy = np.sum(dc[third:2 * third][pf[third:2 * third] == 1])
        omega = mid_buy

        # 尾盤(後1/3): 突破動能 (正值)
        late_ret = dp[-1] - dp[2 * third]
        late_buy = np.sum(dc[2 * third:][pf[2 * third:] == 1])
        U = max(0, late_ret) * late_buy  # 正值

        # 三因子乘積
        raw_val = omega * abs(D) * U
        raw_val = np.log1p(raw_val)  # log compress

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
