"""
F46-F50: Daily + Tick 跨頻率特徵 (批次2)
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (load_tick_data, load_daily_data, compute_tick_daily_intermediates,
                   rolling_zscore, post_process)


def _merge_tick_daily(stock_id):
    df_tick = load_tick_data(stock_id)
    if df_tick.empty:
        return pd.DataFrame()
    tick_daily = compute_tick_daily_intermediates(df_tick)
    if tick_daily.empty:
        return pd.DataFrame()

    df_daily = load_daily_data(stock_id)
    if df_daily.empty:
        return tick_daily

    daily_cols = ['Date', 'StockId']
    optional_cols = ['收盤價', '開盤價', '成交量(千股)', '報酬率']
    for c in optional_cols:
        if c in df_daily.columns:
            daily_cols.append(c)

    df_daily_sel = df_daily[daily_cols].copy()
    merged = tick_daily.merge(df_daily_sel, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    if 'StockId' not in merged.columns:
        merged['StockId'] = stock_id
    return merged


def compute_f46_helical_turbulence_filter(stock_id):
    """高螺旋度壓制耗散 - 大單/碎單信雜比 × 正向漲幅"""
    FEATURE_NAME = 'f_afd_helical_turbulence_filter'
    merged = _merge_tick_daily(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    signal = merged['large_buy'].values.astype(np.float64)
    noise = merged['small_buy'].values.astype(np.float64) + 1.0
    delta_p = np.maximum(0, merged['close_p'].values - merged['open_p'].values)
    raw = (signal / noise) * delta_p
    raw = np.log1p(raw)

    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f47_occlusion_zone_vorticity(stock_id):
    """錮囚區渦度 - 廣大買盤支撐 vs 高價區"""
    FEATURE_NAME = 'f_afd_occlusion_zone_vorticity'
    merged = _merge_tick_daily(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    f_broad = merged['above_vwap_buy'].values.astype(np.float64)
    f_narrow = merged['above_vwap_high_buy'].values.astype(np.float64)
    delta_close = merged['close_p'].values - merged['open_p'].values

    raw = np.maximum(0, f_broad - f_narrow) * delta_close
    raw = np.log1p(np.abs(raw)) * np.sign(raw)

    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f48_ground_relative_friction_shear(stock_id):
    """大盤摩擦剪切 - 個股超額報酬 × 大盤 × Vol"""
    FEATURE_NAME = 'f_afd_ground_relative_friction_shear'
    merged = _merge_tick_daily(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    if '報酬率' in merged.columns:
        stock_ret = merged['報酬率'].fillna(0).values
    else:
        stock_ret = ((merged['close_p'] - merged['open_p']) / (merged['open_p'] + 1e-8)).values

    # 大盤報酬近似: 使用個股日級報酬的 20 日 rolling mean
    market_ret = pd.Series(stock_ret).rolling(20).mean().fillna(0).values
    v_micro = np.maximum(0, stock_ret - market_ret)
    vol = merged['total_vol'].values.astype(np.float64)

    raw = market_ret * v_micro * vol
    raw = np.log1p(np.abs(raw)) * np.sign(raw)

    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f49_optimal_drag_coefficient(stock_id):
    """最佳表面阻力係數 - 高斯核篩選"""
    FEATURE_NAME = 'f_afd_optimal_drag_coefficient'
    df_tick = load_tick_data(stock_id)
    if df_tick.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    records = []
    for date_int, grp in df_tick.groupby('Date_int'):
        if len(grp) < 10:
            continue
        dp = grp['DealPrice'].values.astype(np.float64)
        dc = grp['DealCount'].values.astype(np.float64)
        sp = grp['SellPr'].values
        bp = grp['BuyPr'].values

        spread_mean = np.mean(sp - bp)
        vol = np.sum(dc)
        drag_ratio = spread_mean / (vol + 1e-10)

        # OLS slope
        x = np.arange(len(dp), dtype=np.float64)
        slope = np.polyfit(x, dp, 1)[0] if len(dp) > 2 else 0.0

        records.append({
            'Date': date_int,
            'drag_ratio': drag_ratio,
            'slope': slope,
            'StockId': stock_id,
        })

    if not records:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    df = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)
    # 高斯核: optimal = 20日 median
    optimal = df['drag_ratio'].rolling(20, min_periods=5).median().fillna(df['drag_ratio'])
    sigma = df['drag_ratio'].rolling(20, min_periods=5).std().fillna(1e-10) + 1e-10
    kernel = np.exp(-((df['drag_ratio'] - optimal) ** 2) / (2 * sigma ** 2))
    raw = kernel * np.maximum(0, df['slope'])
    df[FEATURE_NAME] = post_process(rolling_zscore(raw, window=42))
    return df[['StockId', 'Date', FEATURE_NAME]]


def compute_f50_friction_riverbend_exchange(stock_id):
    """河彎效應 - spread不平衡 × 主動買量 × 正向ΔP"""
    FEATURE_NAME = 'f_afd_friction_riverbend_exchange'
    df_tick = load_tick_data(stock_id)
    if df_tick.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    records = []
    for date_int, grp in df_tick.groupby('Date_int'):
        n = len(grp)
        if n < 50:
            continue
        dp = grp['DealPrice'].values.astype(np.float64)
        dc = grp['DealCount'].values.astype(np.float64)
        pf = grp['PrFlag'].values
        sp = grp['SellPr'].values
        bp = grp['BuyPr'].values

        BIN = 50
        n_bins = n // BIN
        total_val = 0.0
        for i in range(n_bins):
            s, e = i * BIN, (i + 1) * BIN
            spread = sp[s:e] - bp[s:e]
            # V_cross = spread position shift (lag=5 within bin)
            v_cross = np.mean(np.diff(spread[:min(5, len(spread))])) if len(spread) > 1 else 0.0
            # V_stream = net buy
            direction = np.where(pf[s:e] == 1, 1.0, np.where(pf[s:e] == 0, -1.0, 0.0))
            v_stream = np.sum(dc[s:e] * direction)
            delta_p = max(0, dp[e - 1] - dp[s])
            total_val += v_cross * v_stream * delta_p

        records.append({
            'Date': date_int,
            'raw': total_val,
            'StockId': stock_id,
        })

    if not records:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    df = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)
    df[FEATURE_NAME] = post_process(rolling_zscore(df['raw'], window=20))
    return df[['StockId', 'Date', FEATURE_NAME]]


if __name__ == '__main__':
    import time
    fns = [
        ('F46', compute_f46_helical_turbulence_filter),
        ('F47', compute_f47_occlusion_zone_vorticity),
        ('F48', compute_f48_ground_relative_friction_shear),
        ('F49', compute_f49_optimal_drag_coefficient),
        ('F50', compute_f50_friction_riverbend_exchange),
    ]
    for label, fn in fns:
        for sid in ['2330']:
            t0 = time.time()
            result = fn(sid)
            print(f"{label} {sid}: {len(result)} days, {time.time()-t0:.1f}s")
