"""
F31-F36: Broker+Tick 跨資料特徵 (批次2)
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (load_tick_data, load_broker_data, preprocess_broker_daily,
                   rolling_zscore, post_process)


def _merge_tick_broker(stock_id):
    df_tick = load_tick_data(stock_id)
    df_broker = load_broker_data(stock_id)
    if df_tick.empty or df_broker.empty:
        return pd.DataFrame()

    tick_daily = df_tick.groupby('Date_int').agg(
        open_p=('DealPrice', 'first'),
        close_p=('DealPrice', 'last'),
        totalVol=('DealCount', 'sum')
    ).reset_index().rename(columns={'Date_int': 'Date'})
    tick_daily['ret'] = (tick_daily['close_p'] - tick_daily['open_p']) / (tick_daily['open_p'] + 1e-8)

    broker_daily = preprocess_broker_daily(df_broker)
    if broker_daily.empty:
        return pd.DataFrame()

    merged = tick_daily.merge(broker_daily, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    merged['StockId'] = stock_id
    return merged


def compute_f31_nlsv_energy_extraction(stock_id):
    """大戶淨買與報酬的能量提取矩陣行列式"""
    FEATURE_NAME = 'f_afd_nlsv_energy_extraction'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 25:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    top10 = merged['Top10_NetBuy'].values.astype(np.float64)
    ret = merged['ret'].values

    raw_vals = np.zeros(len(merged))
    for i in range(20, len(merged)):
        # 5日和20日斜率
        s5_t = np.polyfit(np.arange(5), top10[i-5:i], 1)[0] if np.std(top10[i-5:i]) > 0 else 0
        s5_r = np.polyfit(np.arange(5), ret[i-5:i], 1)[0] if np.std(ret[i-5:i]) > 0 else 0
        s20_t = np.polyfit(np.arange(20), top10[i-20:i], 1)[0] if np.std(top10[i-20:i]) > 0 else 0
        s20_r = np.polyfit(np.arange(20), ret[i-20:i], 1)[0] if np.std(ret[i-20:i]) > 0 else 0
        det_e = s5_t * s20_r - s5_r * s20_t
        dir_val = max(0, merged['close_p'].iloc[i] - merged['open_p'].iloc[i])
        raw_vals[i] = det_e * dir_val

    merged['raw'] = raw_vals
    merged[FEATURE_NAME] = post_process(rolling_zscore(merged['raw'], window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f32_entrainment_mass_flux(stock_id):
    """夾帶率低時觸發 - exp(-ε) × ΔP"""
    FEATURE_NAME = 'f_afd_entrainment_mass_flux'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 10:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    top10_buy = merged['TotalBuy'].values * 0.3  # rough top10 estimate
    retail_buy = merged['TotalBuy'].values * 0.7
    # ε = 散戶/大戶 比率的 5 日梯度
    ratio = retail_buy / (np.abs(merged['Top10_NetBuy'].values) + 1)
    epsilon = pd.Series(ratio).rolling(5).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 5 else 0, raw=False).fillna(0).values

    delta_p = np.maximum(0, merged['close_p'].values - merged['open_p'].values)
    raw = delta_p * np.exp(-np.clip(epsilon, -5, 5))
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f33_dynamic_vppgf_asymmetry(stock_id):
    """大戶動量平方 vs 散戶動量平方"""
    FEATURE_NAME = 'f_afd_dynamic_vppgf_asymmetry'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    zeta_large = merged['Top10_NetBuy'].values.astype(np.float64)
    zeta_small = merged['Retail_NetBuy'].values.astype(np.float64)
    delta_close = merged['close_p'].values - merged['open_p'].values

    raw = np.maximum(0, zeta_large**2 - zeta_small**2) * delta_close
    raw = np.log1p(np.abs(raw)) * np.sign(raw)
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f34_cold_pool_buoyancy_gradient(stock_id):
    """大戶動力抬升克服散戶負浮力"""
    FEATURE_NAME = 'f_afd_cold_pool_buoyancy_gradient'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    u_dyn = np.abs(merged['Top5_NetBuy'].values.astype(np.float64))
    b_cold = np.abs(np.minimum(0, merged['Retail_NetBuy'].values.astype(np.float64)))
    dir_sign = np.sign(merged['close_p'].values - merged['open_p'].values)

    raw = np.maximum(0, u_dyn - b_cold) * dir_sign
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f35_goldilocks_cold_pool(stock_id):
    """適居帶冷池 - 適度回檔後的噴出"""
    FEATURE_NAME = 'f_afd_goldilocks_cold_pool'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 10:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    retail_cum = merged['Retail_NetBuy'].rolling(5).sum().fillna(0).values
    threshold = pd.Series(np.abs(retail_cum)).rolling(20).quantile(0.75).fillna(1).values
    g_c = retail_cum * np.maximum(0, threshold - np.abs(retail_cum))
    u_draft = merged['Top5_NetBuy'].values.astype(np.float64)
    dir_sign = np.sign(merged['close_p'].values - merged['open_p'].values)

    raw = g_c * u_draft * dir_sign
    raw = np.log1p(np.abs(raw)) * np.sign(raw)
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=42))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f36_updraft_width_entrainment(stock_id):
    """上升氣流寬度 × 正漲幅"""
    FEATURE_NAME = 'f_afd_updraft_width_entrainment'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    w_up = merged['NB_BuyBrokers'].values.astype(np.float64)
    delta_p = np.maximum(0, merged['close_p'].values - merged['open_p'].values)
    raw = w_up * delta_p
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw), window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


if __name__ == '__main__':
    import time
    fns = [
        ('F31', compute_f31_nlsv_energy_extraction),
        ('F32', compute_f32_entrainment_mass_flux),
        ('F33', compute_f33_dynamic_vppgf_asymmetry),
        ('F34', compute_f34_cold_pool_buoyancy_gradient),
        ('F35', compute_f35_goldilocks_cold_pool),
        ('F36', compute_f36_updraft_width_entrainment),
    ]
    for label, fn in fns:
        for sid in ['2330']:
            t0 = time.time()
            result = fn(sid)
            print(f"{label} {sid}: {len(result)} days, {time.time()-t0:.1f}s")
