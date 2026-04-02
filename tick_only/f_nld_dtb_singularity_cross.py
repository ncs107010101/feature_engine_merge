"""
特徵名: f_nld_dtb_singularity_cross
靈感來源: DTB 奇點的 3D 相空間展開
計算邏輯: 3D (ΔP, d(ΔP)/dt, d²(ΔP)/dt²) → 相鄰外積 → Z 軸分量 × sign(ΔVol)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (cross_product_3d_z, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_dtb_singularity_cross'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    if len(prices) < 30:
        return 0.0
    
    dp = np.diff(prices)
    # 一階速度: EWM 平滑
    dp_s = pd.Series(dp).ewm(span=5, min_periods=1).mean().values
    # 二階加速度
    ddp = np.diff(dp_s)
    # 截取到相同長度
    n = min(len(dp) - 1, len(ddp))
    if n < 10:
        return 0.0
    
    x = dp[1:n+1]      # ΔP
    y = dp_s[1:n+1]     # 速度 (平滑後)
    z = ddp[:n]          # 加速度
    
    # 相鄰外積 Z 軸分量
    z_comp = cross_product_3d_z(
        x[:-1], y[:-1], z[:-1],
        x[1:], y[1:], z[1:]
    )
    
    # 乘以 sign(ΔVol) 作為不對稱符號
    dv = np.diff(volumes)
    dv_sign = np.sign(dv[2:n+1])
    min_len = min(len(z_comp), len(dv_sign))
    
    if min_len == 0:
        return 0.0
    
    weighted = z_comp[:min_len] * dv_sign[:min_len]
    return np.mean(weighted)


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
