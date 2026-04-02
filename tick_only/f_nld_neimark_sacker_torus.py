"""
特徵名: f_nld_neimark_sacker_torus
靈感來源: Neimark-Sacker 分岔 — 不變環面碰撞
計算邏輯: 3D (VWAP, ΔP, ΔVol) → 外積法向量 → 二階差分 × sign(ΔP)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_neimark_sacker_torus'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    if len(prices) < 30:
        return 0.0
    
    # VWAP
    cum_qty = np.cumsum(volumes)
    cum_pq = np.cumsum(prices * volumes)
    vwap = cum_pq / (cum_qty + 1e-10)
    
    dp = np.diff(prices)
    dv = np.diff(volumes)
    vwap_mid = vwap[1:]  # 對齊 diff
    
    n = min(len(dp), len(dv), len(vwap_mid))
    if n < 10:
        return 0.0
    
    # 標準化
    x = vwap_mid[:n] / (np.std(vwap_mid[:n]) + 1e-10)
    y = dp[:n] / (np.std(dp[:n]) + 1e-10)
    z = dv[:n] / (np.std(dv[:n]) + 1e-10)
    
    # 計算相鄰向量外積的 norm
    # 外積: v1 × v2 → ||cross|| = sqrt(cx² + cy² + cz²)
    cx = y[:-1]*z[1:] - z[:-1]*y[1:]
    cy = z[:-1]*x[1:] - x[:-1]*z[1:]
    cz = x[:-1]*y[1:] - y[:-1]*x[1:]
    
    norm = np.sqrt(cx**2 + cy**2 + cz**2)
    
    if len(norm) < 3:
        return 0.0
    
    # 二階差分
    d2_norm = np.diff(norm, n=2)
    mean_d2 = np.mean(d2_norm)
    
    if not np.isfinite(mean_d2):
        return 0.0
    
    asym = np.sign(prices[-1] - prices[0])
    return mean_d2 * asym


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
