"""
特徵名: f_nld_inhibitory_collapse_accel
靈感來源: 抑制性可塑性崩潰 — E/I 比率的二階加速度
計算邏輯: C_EI = I_Vol/(E_Vol+ε) → log → d² → 負加速 × sign(ΔP)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_inhibitory_collapse_accel'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    prices = day_ticks['DealPrice'].values
    time_sec = day_ticks['DealTimeSecond'].values
    if len(volumes) < 50:
        return 0.0
    
    # 按 30 秒 bin 計算 E/I 比率
    bin_size = 30
    t_min = time_sec.min()
    bins = (time_sec - t_min) // bin_size
    unique_bins = np.unique(bins)
    
    if len(unique_bins) < 5:
        return 0.0
    
    ratios = []
    for b in unique_bins:
        mask = bins == b
        e_vol = np.sum(volumes[mask & (prflag == 1)]) + 1
        i_vol = np.sum(volumes[mask & (prflag == 0)]) + 1
        ratios.append(np.log(i_vol / e_vol))
    
    ratio_series = pd.Series(ratios)
    d2 = safe_second_derivative(ratio_series, ewm_span=3)
    
    d2_vals = d2.dropna().values
    if len(d2_vals) == 0:
        return 0.0
    
    # 負加速度 = 抑制力崩潰
    neg_accel = np.mean(np.minimum(d2_vals, 0))
    
    if not np.isfinite(neg_accel):
        return 0.0
    
    asym = np.sign(prices[-1] - prices[0])
    return neg_accel * asym


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
