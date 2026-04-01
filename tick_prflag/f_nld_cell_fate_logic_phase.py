"""
特徵名: f_nld_cell_fate_logic_phase
靈感來源: p53 AND 邏輯閘 — 買賣加速度的相位角
計算邏輯: arctan2(d²V_sell/dt², d²V_buy/dt²) → cos(φ) × AsymSign
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_cell_fate_logic_phase'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    prices = day_ticks['DealPrice'].values
    if len(volumes) < 50:
        return 0.0
    
    buy_vol = np.where(prflag == 1, volumes, 0).astype(float)
    sell_vol = np.where(prflag == 0, volumes, 0).astype(float)
    
    buy_s = pd.Series(buy_vol).ewm(span=20, min_periods=1).mean()
    sell_s = pd.Series(sell_vol).ewm(span=20, min_periods=1).mean()
    
    d2_buy = safe_second_derivative(buy_s, ewm_span=5)
    d2_sell = safe_second_derivative(sell_s, ewm_span=5)
    
    valid = np.isfinite(d2_buy.values) & np.isfinite(d2_sell.values)
    if np.sum(valid) == 0:
        return 0.0
    
    phi = np.arctan2(d2_sell.values[valid], d2_buy.values[valid])
    cos_phi = np.cos(phi)
    mean_cos = np.mean(cos_phi)
    
    if not np.isfinite(mean_cos):
        return 0.0
    
    asym = np.sign(prices[-1] - prices[0])
    return mean_cos * asym


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
