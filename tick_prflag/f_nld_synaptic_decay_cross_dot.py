"""
特徵名: f_nld_synaptic_decay_cross_dot
靈感來源: 突觸衰減時間常數差異 — 快/慢動態的交叉內積
計算邏輯: E快速平滑 vs I慢速平滑 → 各自二階導 → 內積 × AsymSign
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_synaptic_decay_cross_dot'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    prices = day_ticks['DealPrice'].values
    if len(volumes) < 50:
        return 0.0
    
    # E (外盤/主動買) 成交量序列
    e_vol = np.where(prflag == 1, volumes, 0).astype(float)
    i_vol = np.where(prflag == 0, volumes, 0).astype(float)
    
    # 快速平滑 E (span=5), 慢速平滑 I (span=20)
    e_fast = pd.Series(e_vol).ewm(span=5, min_periods=1).mean()
    i_slow = pd.Series(i_vol).ewm(span=20, min_periods=1).mean()
    
    # 各自二階導數
    d2e = safe_second_derivative(e_fast, ewm_span=5)
    d2i = safe_second_derivative(i_slow, ewm_span=5)
    
    # 內積
    valid = np.isfinite(d2e.values) & np.isfinite(d2i.values)
    if np.sum(valid) == 0:
        return 0.0
    
    dot = np.mean(d2e.values[valid] * d2i.values[valid])
    
    if not np.isfinite(dot):
        return 0.0
    
    asym = np.sign(prices[-1] - prices[0])
    return dot * asym


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
