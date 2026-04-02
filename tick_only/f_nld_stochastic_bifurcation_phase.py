"""
特徵名: f_nld_stochastic_bifurcation_phase
靈感來源: 基因網路隨機分岔 — 雙峰模式向量 · VWAP 加速度內積
計算邏輯: KDE 雙峰 → 模式向量 → 與 VWAP 加速度向量內積
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (find_bimodal_peaks, safe_second_derivative,
                          dot_product_2d, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_stochastic_bifurcation_phase'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    if len(prices) < 50:
        return 0.0
    
    # KDE 雙峰
    peak_high, peak_low = find_bimodal_peaks(prices)
    mode_vec = np.array([peak_high, peak_low])
    
    # VWAP 計算 (累積)
    qty = day_ticks['DealCount'].values.astype(float)
    cum_qty = np.cumsum(qty)
    cum_pq = np.cumsum(prices * qty)
    vwap = cum_pq / (cum_qty + 1e-10)
    
    # VWAP 分買賣近似: 上半/下半日 VWAP
    mid = len(vwap) // 2
    if mid < 5:
        return 0.0
    
    vwap_series = pd.Series(vwap)
    d2_vwap = safe_second_derivative(vwap_series, ewm_span=20)
    
    # 加速度向量 = (前半均值加速度, 後半均值加速度)
    first_half_accel = d2_vwap.iloc[:mid].mean()
    second_half_accel = d2_vwap.iloc[mid:].mean()
    
    if not (np.isfinite(first_half_accel) and np.isfinite(second_half_accel)):
        return 0.0
    
    accel_vec = np.array([first_half_accel, second_half_accel])
    
    # 內積
    result = np.dot(mode_vec, accel_vec)
    return result


def compute_feature(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock_id = df_stock['StockId'].iloc[0]
    results = []
    for date, grp in df_stock.groupby('Date'):
        val = _compute_day(grp)
        if not np.isfinite(val):
            val = 0.0
        results.append({'StockId': stock_id, 'Date': date, 'raw': val})
    
    result_df = pd.DataFrame(results)
    if len(result_df) == 0:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    result_df[FEATURE_NAME] = rolling_zscore(result_df['raw'], 20)
    result_df[FEATURE_NAME] = safe_clip_fillna(result_df[FEATURE_NAME])
    return result_df[['StockId', 'Date', FEATURE_NAME]]
