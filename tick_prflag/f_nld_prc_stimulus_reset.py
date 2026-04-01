"""
特徵名: f_nld_prc_stimulus_reset
靈感來源: 心肌細胞 PRC 相位重置 — 大單刺激引發的加速度重置
計算邏輯: 大單(>95th)打入時 d²VWAP/dt² × sign(大單方向) 條件觸發型
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_prc_stimulus_reset'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    if len(prices) < 50:
        return 0.0
    
    # VWAP
    cum_qty = np.cumsum(volumes)
    cum_pq = np.cumsum(prices * volumes)
    vwap = cum_pq / (cum_qty + 1e-10)
    
    # VWAP 二階導數
    vwap_series = pd.Series(vwap)
    d2_vwap = safe_second_derivative(vwap_series, ewm_span=10)
    d2_vals = d2_vwap.values
    
    # 大單閾值 (95th percentile)
    q95 = np.percentile(volumes, 95)
    large_mask = volumes >= q95
    
    if np.sum(large_mask) == 0:
        return 0.0
    
    # 大單發生時的加速度 × 方向
    large_indices = np.where(large_mask)[0]
    stimulus_values = []
    
    for idx in large_indices:
        if idx < len(d2_vals) and np.isfinite(d2_vals[idx]):
            direction = 1.0 if prflag[idx] == 1 else -1.0
            stimulus_values.append(d2_vals[idx] * direction)
    
    if len(stimulus_values) == 0:
        return 0.0
    
    return np.mean(stimulus_values)


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
