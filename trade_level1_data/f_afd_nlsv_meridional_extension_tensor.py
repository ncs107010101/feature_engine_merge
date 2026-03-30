"""
F16: f_afd_nlsv_meridional_extension_tensor
靈感: NLSV 經向拉長防止波干擾 - 時空域張量跡
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_nlsv_meridional_extension_tensor'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        n = len(df_day)
        if n < 10:
            continue
        dp = df_day['DealPrice'].values.astype(np.float64)
        ts = df_day['DealTimeSecond'].values.astype(np.float64)

        # T_flow = diff(time), P_dist = diff(price)
        t_flow = np.diff(ts, prepend=ts[0]).astype(np.float64)
        p_dist = np.diff(dp, prepend=dp[0])

        # Trace = T_flow × P_dist (mixed term captures time-space extension)
        trace_val = t_flow * p_dist
        # 條件: max(0, p_dist) 僅保留正向延展
        weighted = trace_val * np.maximum(0, p_dist)
        raw_val = np.sum(weighted)

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
