"""
F26: f_afd_moist_cisk_feedback
靈感: CISK 低層輻合與對流加熱反饋
大戶吸籌與價格推升的跨週期共振
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (load_tick_data, load_broker_data, preprocess_broker_daily,
                   rolling_zscore, post_process)

FEATURE_NAME = 'f_afd_moist_cisk_feedback'


def compute_single_stock(stock_id):
    df_tick = load_tick_data(stock_id)
    df_broker = load_broker_data(stock_id)
    if df_tick.empty or df_broker.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    # Tick -> 日級: Close-Open return
    tick_daily = df_tick.groupby('Date_int').agg(
        open_p=('DealPrice', 'first'),
        close_p=('DealPrice', 'last')
    ).reset_index().rename(columns={'Date_int': 'Date'})
    tick_daily['ret'] = (tick_daily['close_p'] - tick_daily['open_p']) / (tick_daily['open_p'] + 1e-8)

    # Broker -> 日級
    broker_daily = preprocess_broker_daily(df_broker)
    if broker_daily.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    # Merge
    merged = tick_daily.merge(broker_daily, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    if len(merged) < 10:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    # 5日滾動: det(S_price^T × S_broker) 近似
    # 使用 rolling covariance 作為近似
    ret = merged['ret'].values
    large_net = merged['Top10_NetBuy'].values.astype(np.float64)
    retail_net = merged['Retail_NetBuy'].values.astype(np.float64)

    raw_vals = np.zeros(len(merged))
    for i in range(5, len(merged)):
        r_slice = ret[i - 5:i]
        l_slice = large_net[i - 5:i]
        # 2×2 matrix: [[cov(ret,ret), cov(ret,large)], [cov(large,ret), cov(large,large)]]
        cov_rr = np.cov(r_slice, l_slice) if np.std(r_slice) > 1e-10 and np.std(l_slice) > 1e-10 else np.eye(2) * 1e-10
        if cov_rr.shape == (2, 2):
            det_val = np.linalg.det(cov_rr)
        else:
            det_val = 0.0
        net_diff = max(0, large_net[i] - retail_net[i])
        raw_vals[i] = det_val * net_diff

    merged['raw'] = raw_vals
    merged['raw'] = np.log1p(np.abs(merged['raw'])) * np.sign(merged['raw'])
    merged[FEATURE_NAME] = post_process(rolling_zscore(merged['raw'], window=42))
    merged['StockId'] = stock_id

    return merged[['StockId', 'Date', FEATURE_NAME]]


if __name__ == '__main__':
    import time
    for sid in ['2330', '1101']:
        t0 = time.time()
        result = compute_single_stock(sid)
        print(f"{sid}: {len(result)} days, {time.time()-t0:.1f}s")
        if not result.empty:
            print(result.describe())
