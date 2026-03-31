"""
F41-F45: Daily + Tick 跨頻率特徵 (批次1)
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

    # 選取需要的 daily 欄位
    daily_cols = ['Date', 'StockId']
    optional_cols = ['收盤價', '開盤價', '成交量(千股)', '報酬率', '融券餘額(張)']
    for c in optional_cols:
        if c in df_daily.columns:
            daily_cols.append(c)

    df_daily_sel = df_daily[daily_cols].copy()
    merged = tick_daily.merge(df_daily_sel, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    if 'StockId' not in merged.columns:
        merged['StockId'] = stock_id
    return merged


def compute_f41_cold_sector_convective_trigger(stock_id):
    """冷區對流加熱: 冷區主動買量 × 冷區斜率"""
    FEATURE_NAME = 'f_afd_cold_sector_convective_trigger'
    merged = _merge_tick_daily(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    raw = merged['cold_zone_buy'] * merged['cold_slope']
    merged[FEATURE_NAME] = post_process(rolling_zscore(raw, window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f42_latent_heat_waterfall_slope(stock_id):
    """融券燃料點火: 融券 × 正向tick斜率"""
    FEATURE_NAME = 'f_afd_latent_heat_waterfall_slope'
    merged = _merge_tick_daily(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    if '融券餘額(張)' not in merged.columns:
        merged['融券餘額(張)'] = 0

    s_short = merged['融券餘額(張)'].fillna(0).values.astype(np.float64)
    p_slope = merged['tick_slope'].values
    # 條件觸發
    trigger = (s_short > 0) & (p_slope > 0)
    raw = pd.Series(np.where(trigger, s_short * p_slope, 0.0))
    merged[FEATURE_NAME] = post_process(rolling_zscore(raw, window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f43_baroclinic_ageostrophic_div(stock_id):
    """前沿價格脫離重心的發散"""
    FEATURE_NAME = 'f_afd_baroclinic_ageostrophic_div'
    merged = _merge_tick_daily(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    raw = np.maximum(0, merged['close_p'].values - merged['vwap'].values) * merged['total_vol'].values
    raw = np.log1p(raw)
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f44_pv_pt_resonance_jacobian(stock_id):
    """宏觀(VWAP)與微觀(tick)斜率的共振 Jacobian"""
    FEATURE_NAME = 'f_afd_pv_pt_resonance_jacobian'
    merged = _merge_tick_daily(stock_id)
    if merged.empty or len(merged) < 10:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    vwap_slope = merged['vwap'].diff(5).fillna(0).values / (merged['vwap'].shift(5).fillna(merged['vwap']).values + 1e-8)
    tick_slope = merged['tick_slope'].values

    # det(M) ≈ vwap_slope * tick_slope² - vwap_slope² * tick_slope
    det_m = vwap_slope * tick_slope**2 - vwap_slope**2 * tick_slope
    dir_val = np.maximum(0, merged['close_p'].values - merged['vwap'].values)
    raw = det_m * dir_val

    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f45_storm_relative_helicity(stock_id):
    """風暴相對螺旋度"""
    FEATURE_NAME = 'f_afd_storm_relative_helicity'
    merged = _merge_tick_daily(stock_id)
    if merged.empty or len(merged) < 10:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    v_srf = merged['tick_mean_price'].values - merged['vwap'].values
    if '報酬率' in merged.columns:
        s_env = merged['報酬率'].rolling(5).sum().fillna(0).values
    else:
        s_env = merged['close_p'].pct_change(5).fillna(0).values

    cross = np.abs(v_srf * s_env)  # 2D 外積範數近似
    dot = v_srf * s_env
    raw = cross * np.sign(dot)

    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


if __name__ == '__main__':
    import time
    fns = [
        ('F41', compute_f41_cold_sector_convective_trigger),
        ('F42', compute_f42_latent_heat_waterfall_slope),
        ('F43', compute_f43_baroclinic_ageostrophic_div),
        ('F44', compute_f44_pv_pt_resonance_jacobian),
        ('F45', compute_f45_storm_relative_helicity),
    ]
    for label, fn in fns:
        for sid in ['2330']:
            t0 = time.time()
            result = fn(sid)
            print(f"{label} {sid}: {len(result)} days, {time.time()-t0:.1f}s")
