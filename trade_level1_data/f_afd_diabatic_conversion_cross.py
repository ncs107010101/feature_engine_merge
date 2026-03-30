"""
F01: f_afd_diabatic_conversion_cross
靈感: DRV 非絕熱加熱轉換率
外積衡量主動單方向與價格位移正交時的「潛熱釋放」，區分真假突破
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature, get_all_stock_ids

FEATURE_NAME = 'f_afd_diabatic_conversion_cross'


def compute_single_stock(df_stock):
    """計算單一股票所有日期的特徵"""
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        if len(df_day) < 10:
            continue
        dp = df_day['DealPrice'].values
        dc = df_day['DealCount'].values
        pf = df_day['PrFlag'].values

        # V = 主動買量 - 主動賣量 (逐tick)
        V = np.where(pf == 1, dc, np.where(pf == 0, -dc, 0)).astype(np.float64)
        # ΔP = 價格變動
        delta_p = np.diff(dp, prepend=dp[0])

        # G_E = V × max(0, ΔP) (正向轉換)
        G_E = V * np.maximum(0, delta_p)
        # C_A = V × |ΔP| (全方向耗散)
        C_A = V * np.abs(delta_p)

        raw_val = np.sum(G_E - C_A)

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
    # Test with a few stocks first
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
        print(df_all.head(10))
