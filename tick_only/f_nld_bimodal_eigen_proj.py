"""
特徵名: f_nld_bimodal_eigen_proj
靈感來源: 細胞網路雙峰隨機分岔 — 特徵向量與加速度向量投影
計算邏輯: 5分鐘bin KDE雙峰 → 2×2轉移矩陣 → 特徵向量 · 加速度向量
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (find_bimodal_peaks, build_transition_matrix,
                          stationary_distribution, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_bimodal_eigen_proj'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    time_sec = day_ticks['DealTimeSecond'].values
    if len(prices) < 100:
        return 0.0
    
    # 5 分鐘 bins = 300 秒
    bin_size = 300
    t_min = time_sec.min()
    bins = (time_sec - t_min) // bin_size
    unique_bins = np.unique(bins)
    
    if len(unique_bins) < 3:
        return 0.0
    
    projections = []
    for b in unique_bins:
        mask = bins == b
        p_bin = prices[mask]
        
        if len(p_bin) < 20:
            continue
        
        # 雙峰
        peak_high, peak_low = find_bimodal_peaks(p_bin)
        mid = (peak_high + peak_low) / 2
        
        # 將價格分為高/低兩類 (相對於中點)
        above = p_bin >= mid
        below = ~above
        
        # 構建二態轉移矩陣 (0=High, 1=Low)
        states = np.where(above, 0, 1).astype(int)
        if len(states) < 5:
            continue
        
        tm = build_transition_matrix(states, n_states=2, laplace_smooth=1.0)
        pi = stationary_distribution(tm)
        
        # 加速度: 高價區/低價區的二階差分
        dp = np.diff(p_bin)
        if len(dp) < 3:
            continue
        ddp = np.diff(dp)
        
        accel_high = np.mean(ddp[above[2:]]) if np.sum(above[2:]) > 0 else 0.0
        accel_low = np.mean(ddp[~above[2:]]) if np.sum(~above[2:]) > 0 else 0.0
        
        if not (np.isfinite(accel_high) and np.isfinite(accel_low)):
            continue
        
        proj = np.dot(pi, np.array([accel_high, accel_low]))
        projections.append(proj)
    
    if len(projections) == 0:
        return 0.0
    
    return np.mean(projections)


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
