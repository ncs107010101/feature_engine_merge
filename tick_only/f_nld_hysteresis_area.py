"""
特徵名: f_nld_hysteresis_area
靈感來源: 確定性爆發的遲滯迴圈有向面積
計算邏輯: 50-tick bin 內 (ΔP, ΔVol) 軌跡有向面積 (外積公式)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_hysteresis_area'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    if len(prices) < 50:
        return 0.0
    
    dp = np.diff(prices)
    dv = np.diff(volumes)
    
    # 標準化
    dp_std = np.std(dp) + 1e-10
    dv_std = np.std(dv) + 1e-10
    dp_norm = dp / dp_std
    dv_norm = dv / dv_std
    
    # 50-tick bins
    bin_size = 50
    n_bins = len(dp_norm) // bin_size
    
    if n_bins < 2:
        return 0.0
    
    areas = []
    for i in range(n_bins):
        s = i * bin_size
        e = s + bin_size
        x = dp_norm[s:e]
        y = dv_norm[s:e]
        
        # 有向面積 = 0.5 * Σ(x_i * y_{i+1} - x_{i+1} * y_i)
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        areas.append(area)
    
    return np.mean(areas)


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
