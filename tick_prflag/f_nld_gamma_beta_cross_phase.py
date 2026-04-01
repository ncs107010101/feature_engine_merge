"""
特徵名: f_nld_gamma_beta_cross_phase
靈感來源: E/I 比例破壞下的跨頻率相位耦合
計算邏輯: 1s/10s VWAP 變動相位差 → d² → × sign(net_ofi)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_gamma_beta_cross_phase'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    time_sec = day_ticks['DealTimeSecond'].values
    if len(prices) < 100:
        return 0.0
    
    # VWAP
    cum_qty = np.cumsum(volumes)
    cum_pq = np.cumsum(prices * volumes)
    vwap = cum_pq / (cum_qty + 1e-10)
    
    t_min = time_sec.min()
    
    # Gamma 振盪器: 1 秒 VWAP 變動
    gamma_bins = (time_sec - t_min) // 1
    # Beta 振盪器: 10 秒 VWAP 變動
    beta_bins = (time_sec - t_min) // 10
    
    # 聚合 Gamma
    unique_g = np.unique(gamma_bins)
    if len(unique_g) < 10:
        return 0.0
    gamma_vwap = []
    for g in unique_g:
        mask = gamma_bins == g
        gamma_vwap.append(np.mean(vwap[mask]))
    gamma_vwap = np.array(gamma_vwap)
    d_gamma = np.diff(gamma_vwap)
    
    # 聚合 Beta 
    unique_b = np.unique(beta_bins)
    if len(unique_b) < 5:
        return 0.0
    beta_vwap = []
    for b in unique_b:
        mask = beta_bins == b
        beta_vwap.append(np.mean(vwap[mask]))
    beta_vwap = np.array(beta_vwap)
    d_beta = np.diff(beta_vwap)
    
    # 對齊: 每 10 個 gamma 對應 1 個 beta
    n_common = min(len(d_gamma) // 10, len(d_beta))
    if n_common < 3:
        return 0.0
    
    gamma_agg = np.array([np.mean(d_gamma[i*10:(i+1)*10]) for i in range(n_common)])
    beta_agg = d_beta[:n_common]
    
    # 相位角
    g_std = np.std(gamma_agg) + 1e-10
    b_std = np.std(beta_agg) + 1e-10
    theta_g = np.arctan2(gamma_agg / g_std, np.arange(n_common) / n_common)
    theta_b = np.arctan2(beta_agg / b_std, np.arange(n_common) / n_common)
    
    phi_diff = theta_g - theta_b
    
    if len(phi_diff) < 3:
        return 0.0
    
    d2_phi = np.diff(phi_diff, n=2)
    mean_d2 = np.mean(d2_phi)
    
    if not np.isfinite(mean_d2):
        return 0.0
    
    # OFI 方向
    buy_vol = np.sum(volumes[prflag == 1])
    sell_vol = np.sum(volumes[prflag == 0])
    ofi_sign = np.sign(buy_vol - sell_vol)
    
    return mean_d2 * ofi_sign


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
