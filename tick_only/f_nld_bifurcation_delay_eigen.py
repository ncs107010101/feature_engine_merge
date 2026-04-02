"""
特徵名: f_nld_bifurcation_delay_eigen
靈感來源: 對稱吸引子附近的分岔延遲 — 特徵值差距極小時的破局
計算邏輯: 15分鐘bin (ΔP, ΔV) 共變異數矩陣 → 特徵值差距 → 最小gap時加速度 × AsymSign
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (eigenvalue_gap, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_bifurcation_delay_eigen'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    time_sec = day_ticks['DealTimeSecond'].values
    
    if len(prices) < 30:
        return 0.0
    
    dp = np.diff(prices)
    dv = np.diff(volumes)
    t = time_sec[1:]
    
    # 15 分鐘 = 900 秒 bins
    bin_size = 900
    t_min = t.min()
    bins = (t - t_min) // bin_size
    unique_bins = np.unique(bins)
    
    if len(unique_bins) < 2:
        return 0.0
    
    eigen_gaps = []
    accels = []
    
    for b in unique_bins:
        mask = bins == b
        dp_bin = dp[mask]
        dv_bin = dv[mask]
        
        if len(dp_bin) < 5:
            continue
        
        # 標準化
        dp_std = np.std(dp_bin) + 1e-10
        dv_std = np.std(dv_bin) + 1e-10
        
        data_mat = np.column_stack([dp_bin / dp_std, dv_bin / dv_std])
        cov = np.cov(data_mat.T)
        gap = eigenvalue_gap(cov)
        
        # 加速度 = 該 bin 內價格的二階差分均值
        if len(dp_bin) >= 3:
            d2p = np.diff(dp_bin, n=1)  # 一階差分的差分
            accel = np.mean(d2p)
        else:
            accel = 0.0
        
        eigen_gaps.append(gap)
        accels.append(accel)
    
    if len(eigen_gaps) == 0:
        return 0.0
    
    # 找特徵值差距最小的 bin (最接近分岔點)
    eigen_gaps = np.array(eigen_gaps)
    accels = np.array(accels)
    min_gap_idx = np.argmin(eigen_gaps)
    
    # 不對稱符號
    close = prices[-1]
    open_ = prices[0]
    asym = np.sign(close - open_)
    
    return accels[min_gap_idx] * asym


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
