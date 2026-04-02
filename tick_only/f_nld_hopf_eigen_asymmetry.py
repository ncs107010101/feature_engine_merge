"""
特徵名: f_nld_hopf_eigen_asymmetry
靈感來源: Hopf 分岔 — 轉移矩陣的不對稱性決定系統方向
計算邏輯: 逐筆 ΔP 離散三態 → 3×3 轉移矩陣 → 主特徵值 × (P(U→U)-P(D→D))
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (build_transition_matrix, leading_eigenvalue_real,
                          price_to_three_states, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_hopf_eigen_asymmetry'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    if len(prices) < 20:
        return 0.0
    dp = np.diff(prices)
    states = price_to_three_states(dp)
    tm = build_transition_matrix(states, n_states=3, laplace_smooth=1.0)
    
    max_ev = leading_eigenvalue_real(tm)
    asym = tm[0, 0] - tm[2, 2]  # P(Up→Up) - P(Down→Down)
    return max_ev * asym


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
