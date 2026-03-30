"""
F17: f_afd_continuous_spectrum_transient
靈感: 擾動投射到連續譜的瞬間非模態增長
超高頻(1-tick)與長週期(60-tick)趨勢的瞬間對齊
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_continuous_spectrum_transient'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 65:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)
        dc = df_day['DealCount'].values.astype(np.float64)

        # S_cont = 1-tick 瞬時跳動
        s_cont = np.diff(dp, n=1, prepend=dp[0])
        # S_disc = 60-tick 趨勢
        s_disc = np.zeros(n, dtype=np.float64)
        s_disc[60:] = dp[60:] - dp[:-60]

        # alignment = max(0, S_cont × S_disc) 同向才觸發
        alignment = np.maximum(0, s_cont * s_disc)
        # 成交量加權
        weighted = alignment * dc
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
