"""
F05: f_afd_vortex_stretching_dot
靈感: 渦度拉伸 - 向上價格與流動性真空梯度的內積
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_vortex_stretching_dot'


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        if len(df_day) < 10:
            continue
        dp = df_day['DealPrice'].values
        bp = df_day['BuyPr'].values
        sp = df_day['SellPr'].values

        # ∇_liq = diff(spread) 流動性梯度
        spread = sp - bp
        grad_liq = np.diff(spread, prepend=spread[0])
        # v = diff(DealPrice) 價格變動
        v = np.diff(dp, prepend=dp[0])

        # stretching = v × grad_liq × sign(max(0, v))
        # 當 v > 0: trigger = 1, 否則 0
        trigger = (v > 0).astype(np.float64)
        stretching = v * grad_liq * trigger

        raw_val = np.sum(stretching)

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
