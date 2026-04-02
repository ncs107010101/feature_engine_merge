"""
特徵名: f_nld_saddle_node_phase_accel
靈感來源: 鞍結分岔 — 成交價相對買賣報價的角加速度
計算邏輯: arctan2(SellPr-DealPrice, DealPrice-BuyPr) → d²φ/dt² 的日內極值
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (phase_angle, safe_second_derivative,
                          rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_saddle_node_phase_accel'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    buy_pr = day_ticks['BuyPr'].values
    sell_pr = day_ticks['SellPr'].values
    
    if len(prices) < 20:
        return 0.0
    
    x = prices - buy_pr   # 成交價離買價的距離
    y = sell_pr - prices   # 賣價離成交價的距離
    
    # 去除零或負值
    x = np.maximum(x, 1e-4)
    y = np.maximum(y, 1e-4)
    
    phi = phase_angle(x, y)
    phi_series = pd.Series(phi)
    
    d2phi = safe_second_derivative(phi_series, ewm_span=10)
    
    # 取日內極端值（最大絕對值，保留符號）
    d2phi_clean = d2phi.dropna()
    if len(d2phi_clean) == 0:
        return 0.0
    
    abs_max_idx = d2phi_clean.abs().idxmax()
    return d2phi_clean.loc[abs_max_idx]


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
