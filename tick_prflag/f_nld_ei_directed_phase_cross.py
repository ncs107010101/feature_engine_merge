"""
特徵名: f_nld_ei_directed_phase_cross
靈感來源: E/I 振盪器有向相位外積
計算邏輯: E/I 成交量 10s bin → 相位角 → 2D 外積
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (cross_product_2d, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_ei_directed_phase_cross'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    time_sec = day_ticks['DealTimeSecond'].values
    if len(volumes) < 30:
        return 0.0
    
    # 10 秒 bins
    bin_size = 10
    t_min = time_sec.min()
    bins = (time_sec - t_min) // bin_size
    
    # 按 bin 聚合 E 和 I 成交量
    unique_bins = np.unique(bins)
    if len(unique_bins) < 5:
        return 0.0
    
    e_vols = []
    i_vols = []
    for b in unique_bins:
        mask = bins == b
        e_vol = np.sum(volumes[mask & (prflag == 1)])
        i_vol = np.sum(volumes[mask & (prflag == 0)])
        e_vols.append(e_vol)
        i_vols.append(i_vol)
    
    e_vols = np.array(e_vols, dtype=float)
    i_vols = np.array(i_vols, dtype=float)
    
    # 差分作為「振盪」
    if len(e_vols) < 3:
        return 0.0
    
    de = np.diff(e_vols)
    di = np.diff(i_vols)
    
    # 相位角
    e_std = np.std(de) + 1e-10
    i_std = np.std(di) + 1e-10
    
    theta_e = np.arctan2(np.sin(de / e_std), np.cos(de / e_std))
    theta_i = np.arctan2(np.sin(di / i_std), np.cos(di / i_std))
    
    # E/I 2D 向量外積
    vec_e_x = np.cos(theta_e)
    vec_e_y = np.sin(theta_e)
    vec_i_x = np.cos(theta_i)
    vec_i_y = np.sin(theta_i)
    
    cross = cross_product_2d(vec_e_x, vec_e_y, vec_i_x, vec_i_y)
    return np.mean(cross)


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
