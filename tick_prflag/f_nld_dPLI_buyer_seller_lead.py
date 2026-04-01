"""
特徵名: f_nld_dPLI_buyer_seller_lead
靈感來源: dPLI 有向相位滯後指數 — 買賣相位領先角加速度
計算邏輯: 買/賣 DealCount 差分 → arctan2 近似相位 → 相位差 d² × AsymSign
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_dPLI_buyer_seller_lead'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    prices = day_ticks['DealPrice'].values
    time_sec = day_ticks['DealTimeSecond'].values
    if len(volumes) < 60:
        return 0.0
    
    # 30 秒 bins
    bin_size = 30
    t_min = time_sec.min()
    bins = (time_sec - t_min) // bin_size
    unique_bins = np.unique(bins)
    
    if len(unique_bins) < 5:
        return 0.0
    
    buy_counts = []
    sell_counts = []
    for b in unique_bins:
        mask = bins == b
        buy_counts.append(np.sum(volumes[mask & (prflag == 1)]))
        sell_counts.append(np.sum(volumes[mask & (prflag == 0)]))
    
    buy_arr = np.array(buy_counts, dtype=float)
    sell_arr = np.array(sell_counts, dtype=float)
    
    # 差分作為振盪信號
    d_buy = np.diff(buy_arr)
    d_sell = np.diff(sell_arr)
    
    if len(d_buy) < 3:
        return 0.0
    
    # 近似相位 (arctan2)
    buy_std = np.std(d_buy) + 1e-10
    sell_std = np.std(d_sell) + 1e-10
    
    phi_buy = np.arctan2(d_buy / buy_std, np.arange(len(d_buy)) / len(d_buy))
    phi_sell = np.arctan2(d_sell / sell_std, np.arange(len(d_sell)) / len(d_sell))
    
    # 相位差
    delta_phi = phi_buy - phi_sell
    
    # 二階差分 (角加速度)
    if len(delta_phi) < 3:
        return 0.0
    
    d2_phi = np.diff(delta_phi, n=2)
    mean_d2 = np.mean(d2_phi)
    
    if not np.isfinite(mean_d2):
        return 0.0
    
    asym = np.sign(prices[-1] - prices[0])
    return mean_d2 * asym


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
