"""
特徵名: f_nld_critical_slowing_phase_accel
靈感來源: 臨界慢化 — AC(1) 與成交量的相位角加速度
計算邏輯: 100-tick bin AC(1) → arctan2(Vol, AC1) → d²φ/dt² × sign(OFI)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (phase_angle, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_critical_slowing_phase_accel'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values if 'PrFlag' in day_ticks.columns else None
    
    if len(prices) < 100:
        return 0.0
    
    dp = np.diff(prices)
    bin_size = 100
    n_bins = len(dp) // bin_size
    
    if n_bins < 3:
        return 0.0
    
    phases = []
    for i in range(n_bins):
        s = i * bin_size
        e = s + bin_size
        dp_bin = dp[s:e]
        vol_bin = volumes[s+1:e+1]  # offset by 1 for diff
        
        # AC(1) of price changes
        if len(dp_bin) < 5:
            continue
        ac1 = np.corrcoef(dp_bin[:-1], dp_bin[1:])[0, 1]
        if not np.isfinite(ac1):
            ac1 = 0.0
        ac1 = np.clip(ac1, -0.99, 0.99)
        
        vol_mean = np.mean(vol_bin) if len(vol_bin) > 0 else 1.0
        
        phi = np.arctan2(vol_mean, ac1)
        phases.append(phi)
    
    if len(phases) < 3:
        return 0.0
    
    phases = np.array(phases)
    # 二階差分 (角加速度)
    d2phi = np.diff(phases, n=2)
    
    mean_d2phi = np.mean(d2phi)
    if not np.isfinite(mean_d2phi):
        return 0.0
    
    # OFI 方向
    if prflag is not None:
        buy_vol = np.sum(volumes[prflag == 1])
        sell_vol = np.sum(volumes[prflag == 0])
        ofi_sign = np.sign(buy_vol - sell_vol)
    else:
        ofi_sign = np.sign(prices[-1] - prices[0])
    
    return mean_d2phi * ofi_sign


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
