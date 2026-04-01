"""
特徵名: f_nld_attractor_change_of_mind
靈感來源: 吸引子網路決策逆轉 — 早盤/尾盤向量外積
計算邏輯: 早盤(前30分) vs 尾盤(後30分) 的 (ΔP, OFI) 向量外積
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (cross_product_2d, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_attractor_change_of_mind'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    time_sec = day_ticks['DealTimeSecond'].values
    if len(prices) < 30:
        return 0.0
    
    # 台灣市場: 09:00-13:30 = 32400-48600 秒
    early_mask = time_sec <= (time_sec.min() + 1800)  # 前 30 分鐘
    late_mask = time_sec >= (time_sec.max() - 1800)    # 後 30 分鐘
    
    if np.sum(early_mask) < 5 or np.sum(late_mask) < 5:
        return 0.0
    
    # 早盤向量
    early_dp = prices[early_mask][-1] - prices[early_mask][0]
    early_buy = np.sum(volumes[early_mask & (prflag == 1)])
    early_sell = np.sum(volumes[early_mask & (prflag == 0)])
    early_ofi = early_buy - early_sell
    
    # 尾盤向量
    late_dp = prices[late_mask][-1] - prices[late_mask][0]
    late_buy = np.sum(volumes[late_mask & (prflag == 1)])
    late_sell = np.sum(volumes[late_mask & (prflag == 0)])
    late_ofi = late_buy - late_sell
    
    # 標準化
    dp_scale = max(abs(early_dp), abs(late_dp), 0.01)
    ofi_scale = max(abs(early_ofi), abs(late_ofi), 1)
    
    # 外積
    cross = cross_product_2d(
        early_dp / dp_scale, early_ofi / ofi_scale,
        late_dp / dp_scale, late_ofi / ofi_scale
    )
    return cross


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
