"""
特徵名: f_nld_rna_velocity_splicing
靈感來源: RNA 速度模型 — 報價(未剪接)與成交(已剪接)的外積
計算邏輯: 報價動能 (dBuyPr, dSellPr) × 成交動能 (ActBuyVol, ActSellVol) 外積 → d²
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (cross_product_2d, safe_second_derivative,
                          rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_rna_velocity_splicing'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    buy_pr = day_ticks['BuyPr'].values
    sell_pr = day_ticks['SellPr'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    time_sec = day_ticks['DealTimeSecond'].values
    if len(buy_pr) < 50:
        return 0.0
    
    # 30 秒 bins
    bin_size = 30
    t_min = time_sec.min()
    bins = (time_sec - t_min) // bin_size
    unique_bins = np.unique(bins)
    
    if len(unique_bins) < 5:
        return 0.0
    
    cross_vals = []
    for b in unique_bins:
        mask = bins == b
        if np.sum(mask) < 2:
            continue
        
        # 未剪接 (報價動能)
        bp = buy_pr[mask]
        sp = sell_pr[mask]
        d_bp = bp[-1] - bp[0]
        d_sp = sp[-1] - sp[0]
        
        # 已剪接 (實際成交動能)
        act_buy = np.sum(volumes[mask & (prflag == 1)])
        act_sell = np.sum(volumes[mask & (prflag == 0)])
        
        # 標準化
        scale_price = max(abs(d_bp), abs(d_sp), 0.01)
        scale_vol = max(act_buy, act_sell, 1)
        
        cross = cross_product_2d(
            d_bp / scale_price, d_sp / scale_price,
            act_buy / scale_vol, act_sell / scale_vol
        )
        cross_vals.append(cross)
    
    if len(cross_vals) < 3:
        return 0.0
    
    cross_series = pd.Series(cross_vals)
    d2 = safe_second_derivative(cross_series, ewm_span=3)
    
    mean_d2 = d2.mean()
    return mean_d2 if np.isfinite(mean_d2) else 0.0


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
