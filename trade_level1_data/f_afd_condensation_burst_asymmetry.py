"""
F03: f_afd_condensation_burst_asymmetry
靈感: 降水凝結引發的位渦相變奇點
三次方運算子放大向上掃盤的極端非對稱方向
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_condensation_burst_asymmetry'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        if len(df_day) < 10:
            continue
        dp = df_day['DealPrice'].values
        dc = df_day['DealCount'].values.astype(np.float64)
        sp = df_day['SellPr'].values

        # C_t = max(0, DealPrice_t - SellPr_{t-1}) (突破賣一價的凝結觸發)
        sell_pr_lag = np.roll(sp, 1)
        sell_pr_lag[0] = sp[0]
        C_t = np.maximum(0, dp - sell_pr_lag)

        # Burst = C_t × DealCount
        burst = C_t * dc

        # Σ(Burst³) - 三次方極度放大極端掃盤
        burst_cubed = burst ** 3
        raw_val = np.sum(burst_cubed)

        # log 壓縮極端值
        raw_val = np.log1p(np.abs(raw_val)) * np.sign(raw_val)

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': raw_val
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(
        rolling_zscore(df_res['raw'], window=20)
    )
    return df_res[['StockId', 'Date', FEATURE_NAME]]


if __name__ == '__main__':
    import time
    test_stocks = ['2330', '1101', '2317']
    for sid in test_stocks:
        t0 = time.time()
        df = load_tick_data(sid)
        if df.empty:
            continue
        result = compute_single_stock(df)
        print(f"{sid}: {len(result)} days, {time.time()-t0:.1f}s")
        print(result.describe())
