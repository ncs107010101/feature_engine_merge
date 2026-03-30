"""
F02: f_afd_latent_heat_phase_locking
靈感: DRV 低層正渦度與中層負渦度的相位鎖定
捕捉買賣單在時間序列上的瞬時同步性（賣單被秒吞噬）
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_latent_heat_phase_locking'
BIN_SIZE = 30  # 30-tick 窗口


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        if len(df_day) < BIN_SIZE * 2:
            continue
        dc = df_day['DealCount'].values.astype(np.float64)
        pf = df_day['PrFlag'].values
        dp = df_day['DealPrice'].values

        # 主動買序列與主動賣序列
        buy_seq = np.where(pf == 1, dc, 0.0)
        sell_seq = np.where(pf == 0, dc, 0.0)

        n = len(dc)
        n_bins = n // BIN_SIZE
        if n_bins < 2:
            continue

        total_val = 0.0
        for i in range(n_bins):
            s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
            omega_low = buy_seq[s:e]
            omega_mid = sell_seq[s:e]
            # 內積: 衡量同步性
            inner_prod = np.dot(omega_low, omega_mid)
            # 價格斜率
            p_slice = dp[s:e]
            x = np.arange(len(p_slice), dtype=np.float64)
            if np.std(x) > 0 and np.std(p_slice) > 0:
                slope = np.polyfit(x, p_slice, 1)[0]
            else:
                slope = 0.0
            # 條件: 僅在斜率為正時觸發
            trigger = np.sign(max(0.0, slope))
            total_val += inner_prod * trigger

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': total_val
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
    all_results = []
    for sid in test_stocks:
        t0 = time.time()
        df = load_tick_data(sid)
        if df.empty:
            continue
        result = compute_single_stock(df)
        all_results.append(result)
        print(f"{sid}: {len(result)} days, {time.time()-t0:.1f}s")

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal: {df_all.shape}")
        print(df_all.describe())
