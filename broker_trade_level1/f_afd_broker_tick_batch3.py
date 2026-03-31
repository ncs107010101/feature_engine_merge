"""
F37-F40: Broker+Tick 跨資料特徵 (批次3)
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (load_tick_data, load_broker_data, preprocess_broker_daily,
                   rolling_zscore, post_process)


def _get_tick_daily(stock_id):
    """載入 tick 資料並計算日級中間量"""
    df_tick = load_tick_data(stock_id)
    if df_tick.empty:
        return pd.DataFrame()

    # 日級聚合 (含大小單統計)
    records = []
    for date_int, grp in df_tick.groupby('Date_int'):
        dc = grp['DealCount'].values.astype(np.float64)
        pf = grp['PrFlag'].values
        dp = grp['DealPrice'].values
        sp = grp['SellPr'].values
        bp = grp['BuyPr'].values

        q85 = np.percentile(dc, 85) if len(dc) > 10 else dc.max()
        large_buy = np.sum(dc[(dc >= q85) & (pf == 1)])
        small_buy = np.sum(dc[(dc < q85) & (pf == 1)])
        large_sell = np.sum(dc[(dc >= q85) & (pf == 0)])
        small_sell = np.sum(dc[(dc < q85) & (pf == 0)])
        buy_vol = np.sum(dc[pf == 1])
        sell_vol = np.sum(dc[pf == 0])
        spread_mean = np.mean(sp - bp)

        # 價格二階加速度 (近似)
        delta_p = np.diff(dp, prepend=dp[0])
        accel = np.diff(delta_p, prepend=delta_p[0])
        a_price_mean = np.mean(accel)

        records.append({
            'Date': date_int,
            'open_p': dp[0], 'close_p': dp[-1],
            'large_buy': large_buy, 'small_buy': small_buy,
            'large_sell': large_sell, 'small_sell': small_sell,
            'buy_vol': buy_vol, 'sell_vol': sell_vol,
            'spread_mean': spread_mean,
            'a_price': a_price_mean,
            'totalVol': np.sum(dc),
        })

    return pd.DataFrame(records)


def _merge_all(stock_id):
    tick_daily = _get_tick_daily(stock_id)
    if tick_daily.empty:
        return pd.DataFrame()
    df_broker = load_broker_data(stock_id)
    if df_broker.empty:
        return pd.DataFrame()
    broker_daily = preprocess_broker_daily(df_broker)
    if broker_daily.empty:
        return pd.DataFrame()

    merged = tick_daily.merge(broker_daily, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    merged['StockId'] = stock_id
    merged['ret'] = (merged['close_p'] - merged['open_p']) / (merged['open_p'] + 1e-8)
    return merged


def compute_f37_svc_baroclinic_stretching(stock_id):
    """買盤加速度 × 主動買流向"""
    FEATURE_NAME = 'f_afd_svc_baroclinic_stretching'
    merged = _merge_all(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    # A_flow = 主動買量的2階差分
    a_flow = merged['buy_vol'].diff().diff().fillna(0).values
    # ω_s = 主動買 / 總量
    omega_s = merged['buy_vol'].values / (merged['totalVol'].values + 1)
    delta_p = np.maximum(0, merged['close_p'].values - merged['open_p'].values)

    raw = np.maximum(0, a_flow * omega_s) * delta_p
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f38_random_vorticity_stretching(stock_id):
    """散戶與大戶的拓撲拉伸 Jacobian"""
    FEATURE_NAME = 'f_afd_random_vorticity_stretching'
    merged = _merge_all(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    # 大小單方向流
    v_small = (merged['small_buy'] - merged['small_sell']).values.astype(np.float64)
    v_large = (merged['large_buy'] - merged['large_sell']).values.astype(np.float64)
    delta_p = (merged['close_p'] - merged['open_p']).values

    # 日級 Jacobian: det = d(v_small)/dt * d(v_large)/d(dp) - d(v_small)/d(dp) * d(v_large)/dt
    dv_s_dt = np.diff(v_small, prepend=v_small[0])
    dv_l_dt = np.diff(v_large, prepend=v_large[0])
    # d/d(dp) 近似: 用前一日 dp 做差分
    ddp = np.diff(delta_p, prepend=delta_p[0])
    dv_s_dp = dv_s_dt / (ddp + 1e-8)  # rough
    dv_l_dp = dv_l_dt / (ddp + 1e-8)

    det_j = dv_s_dt * dv_l_dp - dv_s_dp * dv_l_dt
    dir_val = np.maximum(0, delta_p)
    raw = det_j * dir_val
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f39_surface_drag_convergence(stock_id):
    """Spread 梯度 × 價格加速度"""
    FEATURE_NAME = 'f_afd_surface_drag_convergence'
    merged = _merge_all(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    # 日級 spread 的 diff (drag gradient)
    drag = pd.Series(merged['spread_mean'].values).diff().fillna(0).values
    a_price = merged['a_price'].values

    raw = np.maximum(0, drag) * np.maximum(0, a_price)
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f40_rfd_occlusion_wrap_around(stock_id):
    """砸盤後正交反轉"""
    FEATURE_NAME = 'f_afd_rfd_occlusion_wrap_around'
    merged = _merge_all(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    # 使用日級近似: 賣壓 × 反轉買壓
    v_rfd = merged['sell_vol'].values.astype(np.float64)
    v_wrap = merged['buy_vol'].values.astype(np.float64)
    dir_sign = np.sign(merged['close_p'].values - merged['open_p'].values)

    # 條件: 高賣壓 + 正向收盤 = RFD 包覆
    sell_z = rolling_zscore(pd.Series(v_rfd), 20).fillna(0).values
    raw = np.maximum(0, sell_z) * v_wrap * dir_sign
    raw = np.log1p(np.abs(raw)) * np.sign(raw)
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


if __name__ == '__main__':
    import time
    fns = [
        ('F37', compute_f37_svc_baroclinic_stretching),
        ('F38', compute_f38_random_vorticity_stretching),
        ('F39', compute_f39_surface_drag_convergence),
        ('F40', compute_f40_rfd_occlusion_wrap_around),
    ]
    for label, fn in fns:
        for sid in ['2330']:
            t0 = time.time()
            result = fn(sid)
            print(f"{label} {sid}: {len(result)} days, {time.time()-t0:.1f}s")
