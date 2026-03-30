"""
F04: f_afd_moist_enstrophy_jacobian
靈感: 濕度異常增加系統總擬能，加速非線性演化
Jacobian 行列式衡量相空間的收縮/擴張
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_tick_data, rolling_zscore, post_process, save_feature

FEATURE_NAME = 'f_afd_moist_enstrophy_jacobian'
BIN_SIZE = 50


def compute_single_stock(df_stock):
    results = []
    for date_int, df_day in df_stock.groupby('Date_int'):
        if len(df_day) < BIN_SIZE * 2:
            continue
        dp = df_day['DealPrice'].values
        tq = df_day['TotalQty'].values.astype(np.float64)
        ts = df_day['DealTimeSecond'].values.astype(np.float64)

        n = len(dp)
        n_bins = n // BIN_SIZE
        if n_bins < 2:
            continue

        total_det = 0.0
        for i in range(n_bins):
            s, e = i * BIN_SIZE, (i + 1) * BIN_SIZE
            p_s, p_e = dp[s], dp[e - 1]
            q_s, q_e = tq[s], tq[e - 1]
            t_s, t_e = ts[s], ts[e - 1]

            dt = t_e - t_s + 1e-6
            dq = q_e - q_s + 1e-6
            d_price = p_e - p_s

            # Jacobian 元素
            dp_dt = d_price / dt       # ∂P/∂t
            dp_dq = d_price / dq       # ∂P/∂Qty
            dq_dt = (q_e - q_s) / dt   # ∂Qty/∂t

            # det(J) = dp_dt * 1 - dp_dq * dq_dt
            det_j = dp_dt - dp_dq * dq_dt

            # 僅在 ∂P/∂t > 0 時保留
            if dp_dt > 0:
                total_det += det_j

        results.append({
            'StockId': df_day['StockId'].iloc[0],
            'Date': date_int,
            'raw': total_det
        })

    if not results:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(['StockId', 'Date']).reset_index(drop=True)
    df_res[FEATURE_NAME] = post_process(
        rolling_zscore(df_res['raw'], window=42)
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
