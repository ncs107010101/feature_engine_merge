"""
特徵名: f_nld_ei_transition_eigen_dot
靈感來源: E/I 狀態轉移矩陣特徵向量 × 價格位移內積
計算邏輯: PrFlag 構建 E/I 2×2 轉移矩陣 → 主特徵向量 · [ΔP_E, ΔP_I]
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (build_transition_matrix, stationary_distribution,
                          prflag_to_ei_states, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_ei_transition_eigen_dot'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    prflag = day_ticks['PrFlag'].values
    if len(prices) < 30:
        return 0.0
    
    ei_states = prflag_to_ei_states(prflag)
    tm = build_transition_matrix(ei_states, n_states=2, laplace_smooth=1.0)
    pi = stationary_distribution(tm)
    
    # 計算 E 和 I 狀態下的價格位移
    dp = np.diff(prices)
    ei_for_dp = ei_states[1:]  # offset by 1
    
    dp_e = dp[ei_for_dp == 0]
    dp_i = dp[ei_for_dp == 1]
    
    mean_dp_e = np.mean(dp_e) if len(dp_e) > 0 else 0.0
    mean_dp_i = np.mean(dp_i) if len(dp_i) > 0 else 0.0
    
    if not (np.isfinite(mean_dp_e) and np.isfinite(mean_dp_i)):
        return 0.0
    
    price_vec = np.array([mean_dp_e, mean_dp_i])
    return np.dot(pi, price_vec)


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
