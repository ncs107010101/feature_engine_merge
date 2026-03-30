"""
F10: f_afd_tropopause_pv_intrusion
靈感: 對流層頂摺疊引發高位渦向下侵入
捕捉砸盤動能被被動買單吸收的「破底翻」信號
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_tropopause_pv_intrusion'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        if len(df_day) < 10:
            continue
        dp = df_day['DealPrice'].values
        dc = df_day['DealCount'].values.astype(np.float64)
        pf = df_day['PrFlag'].values

        delta_p = np.diff(dp, prepend=dp[0])
        # P_drop = min(0, ΔP)
        p_drop = np.minimum(0, delta_p)
        # V_passive_buy = PrFlag=0 的 DealCount (被動買方被打到)
        v_passive_buy = np.where(pf == 0, dc, 0.0)

        # absorption = |P_drop × V_passive_buy|
        absorption = np.abs(p_drop * v_passive_buy)
        total_absorption = np.sum(absorption)

        # 方向: sign(Close - Open)
        close_p = dp[-1]
        open_p = dp[0]
        dir_sign = np.sign(close_p - open_p)
        raw_val = total_absorption * dir_sign

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
