"""
特徵名: f_nld_saddle_loop_divergence
靈感來源: 鞍迴圈分岔 — VWAP 穿越週期的二階發散
計算邏輯: VWAP 穿越間隔序列 → d² × (Price-VWAP) 日內加權均值
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_saddle_loop_divergence'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    if len(prices) < 50:
        return 0.0
    
    # 計算 VWAP
    cum_qty = np.cumsum(volumes)
    cum_pq = np.cumsum(prices * volumes)
    vwap = cum_pq / (cum_qty + 1e-10)
    
    # 找 VWAP 穿越點
    diff_sign = np.sign(prices - vwap)
    crossings = np.where(np.diff(diff_sign) != 0)[0]
    
    if len(crossings) < 4:
        return 0.0
    
    # 穿越間隔序列
    intervals = np.diff(crossings).astype(float)
    
    if len(intervals) < 3:
        return 0.0
    
    # 二階差分
    d2_intervals = np.diff(intervals, n=2)
    
    if len(d2_intervals) == 0:
        return 0.0
    
    # 日內加權偏離
    last_dev = prices[-1] - vwap[-1]
    mean_d2 = np.mean(d2_intervals)
    
    if not np.isfinite(mean_d2):
        return 0.0
    
    return mean_d2 * np.sign(last_dev)


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
