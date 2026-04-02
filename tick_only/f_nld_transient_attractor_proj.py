"""
特徵名: f_nld_transient_attractor_proj
靈感來源: 吸引子捕獲 — 價格加速度投影至漲跌停方向
計算邏輯: d²P/dt² · ((CeilPr-DealPrice) - (DealPrice-FloorPr)) 日內加權均值
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_transient_attractor_proj'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    ceil_pr = day_ticks['CeilPr'].values
    floor_pr = day_ticks['FloorPr'].values
    
    if len(prices) < 20:
        return 0.0
    
    # 價格加速度
    price_series = pd.Series(prices)
    d2p = safe_second_derivative(price_series, ewm_span=10)
    d2p_vals = d2p.values
    
    # 吸引子方向 = (距漲停) - (距跌停)
    # 正值 = 更接近跌停, 負值 = 更接近漲停
    dist_up = ceil_pr - prices
    dist_dn = prices - floor_pr
    attractor_axis = dist_up - dist_dn  # 正 = 遠離漲停/靠近跌停
    
    # 標準化吸引子方向
    scale = (ceil_pr[0] - floor_pr[0]) if (ceil_pr[0] - floor_pr[0]) > 0 else 1.0
    attractor_norm = attractor_axis / scale
    
    # 內積: 加速度 × 吸引子方向
    n = min(len(d2p_vals), len(attractor_norm))
    valid_mask = np.isfinite(d2p_vals[:n])
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    proj = d2p_vals[:n][valid_mask] * attractor_norm[:n][valid_mask]
    return np.mean(proj)


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
