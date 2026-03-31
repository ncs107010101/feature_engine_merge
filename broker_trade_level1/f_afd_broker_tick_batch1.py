"""
F27: f_afd_diabatic_pv_asymmetric
F28: f_afd_jet_exit_radiation
F29: f_afd_edge_wave_phase_locking
F30: f_afd_optimal_growth_hindering

批次定義四個相對簡單的 broker+tick 特徵
每個特徵以函數方式定義，共用 broker 預處理
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (load_tick_data, load_broker_data, preprocess_broker_daily,
                   rolling_zscore, post_process)


def _merge_tick_broker(stock_id):
    """合併 tick 日級與 broker 日級"""
    df_tick = load_tick_data(stock_id)
    df_broker = load_broker_data(stock_id)
    if df_tick.empty or df_broker.empty:
        return pd.DataFrame()

    tick_daily = df_tick.groupby('Date_int').agg(
        open_p=('DealPrice', 'first'),
        close_p=('DealPrice', 'last'),
        buyVol=('DealCount', lambda x: x[df_tick.loc[x.index, 'PrFlag'] == 1].sum()),
        sellVol=('DealCount', lambda x: x[df_tick.loc[x.index, 'PrFlag'] == 0].sum()),
        totalVol=('DealCount', 'sum')
    ).reset_index().rename(columns={'Date_int': 'Date'})
    tick_daily['ret'] = (tick_daily['close_p'] - tick_daily['open_p']) / (tick_daily['open_p'] + 1e-8)

    broker_daily = preprocess_broker_daily(df_broker)
    if broker_daily.empty:
        return pd.DataFrame()

    merged = tick_daily.merge(broker_daily, on='Date', how='inner').sort_values('Date').reset_index(drop=True)
    merged['StockId'] = stock_id
    return merged


def compute_f27_diabatic_pv_asymmetric(stock_id):
    """券商買超2階差分 × 價格報酬"""
    FEATURE_NAME = 'f_afd_diabatic_pv_asymmetric'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    j_buy = merged['Top10_NetBuy'].diff().diff().fillna(0)
    j_sell = merged['Retail_NetBuy'].diff().diff().fillna(0)
    ret = merged['ret']

    raw = np.maximum(0, j_buy) * ret - np.minimum(0, j_sell) * ret
    merged[FEATURE_NAME] = post_process(rolling_zscore(raw, window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f28_jet_exit_radiation(stock_id):
    """大戶淨買 × 價格加速度"""
    FEATURE_NAME = 'f_afd_jet_exit_radiation'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    j = np.maximum(0, merged['Top5_NetBuy'].values.astype(np.float64))
    a_price = merged['ret'].diff().fillna(0).values  # 2nd derivative
    raw = pd.Series(j * a_price)
    merged[FEATURE_NAME] = post_process(rolling_zscore(raw, window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f29_edge_wave_phase_locking(stock_id):
    """大戶與散戶的動量共振 (正=同向，負=背離)"""
    FEATURE_NAME = 'f_afd_edge_wave_phase_locking'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 10:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    # 5日趨勢
    w_top = merged['Top10_NetBuy'].rolling(5).mean().fillna(0)
    w_surf = merged['Retail_NetBuy'].rolling(5).mean().fillna(0)
    dir_sign = np.sign(merged['ret'])

    # 共振強度: 同號=正共振, 異號=負共振; × 報酬方向
    co_movement = w_top * w_surf  # 正=同向, 負=背離
    raw = co_movement * dir_sign
    raw = np.log1p(np.abs(raw)) * np.sign(raw)
    merged[FEATURE_NAME] = post_process(rolling_zscore(pd.Series(raw.values), window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


def compute_f30_optimal_growth_hindering(stock_id):
    """空頭動能被吸收的軋空預測器"""
    FEATURE_NAME = 'f_afd_optimal_growth_hindering'
    merged = _merge_tick_broker(stock_id)
    if merged.empty or len(merged) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])

    h_hinder = np.minimum(0, merged['buyVol'].values - merged['sellVol'].values)
    delta_p = merged['close_p'].values - merged['open_p'].values
    dir_sign = np.sign(delta_p)
    raw = pd.Series(h_hinder * delta_p * dir_sign)
    merged[FEATURE_NAME] = post_process(rolling_zscore(raw, window=20))
    return merged[['StockId', 'Date', FEATURE_NAME]]


if __name__ == '__main__':
    import time
    fns = [
        ('F27', compute_f27_diabatic_pv_asymmetric),
        ('F28', compute_f28_jet_exit_radiation),
        ('F29', compute_f29_edge_wave_phase_locking),
        ('F30', compute_f30_optimal_growth_hindering),
    ]
    for label, fn in fns:
        for sid in ['2330']:
            t0 = time.time()
            result = fn(sid)
            print(f"{label} {sid}: {len(result)} days, {time.time()-t0:.1f}s")
            if not result.empty:
                print(result.describe())
