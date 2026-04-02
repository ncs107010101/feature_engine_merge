"""
特徵名: f_nld_lyapunov_divergence_accel
靈感來源: 非同步率混沌 — Lyapunov 指數的動態替代量
計算邏輯: 相空間 (P, Δt) 軌跡發散距離 → 二階導數 × sign(ΔP)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_lyapunov_divergence_accel'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    time_sec = day_ticks['DealTimeSecond'].values.astype(float)
    if len(prices) < 30:
        return 0.0
    
    # 標準化
    p_std = np.std(prices) + 1e-10
    t_std = np.std(np.diff(time_sec)) + 1e-10
    
    p_norm = prices / p_std
    dt = np.diff(time_sec)
    dt_norm = dt / t_std
    
    # 延遲嵌入 k=10
    k = 10
    if len(p_norm) < k + 20:
        return 0.0
    
    # 計算相空間距離的發散
    n = len(p_norm) - k
    D = np.zeros(n)
    for i in range(n):
        dp = p_norm[i + k] - p_norm[i]
        if i + k < len(dt_norm):
            dt_val = dt_norm[min(i + k - 1, len(dt_norm) - 1)]
        else:
            dt_val = 0
        D[i] = np.sqrt(dp**2 + dt_val**2)
    
    D_series = pd.Series(D)
    d2D = safe_second_derivative(D_series, ewm_span=10)
    
    mean_d2D = d2D.mean()
    if not np.isfinite(mean_d2D):
        return 0.0
    
    # 不對稱符號
    net_dp = prices[-1] - prices[0]
    asym = np.sign(net_dp)
    
    return mean_d2D * asym


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
