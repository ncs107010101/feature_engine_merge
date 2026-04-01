"""
特徵名: f_nld_coupled_oscillator_rotation
靈感來源: 三個耦合振盪器的旋轉流形
計算邏輯: (dBuyPr/dt, dSellPr/dt, dDealPrice/dt) 3D 外積
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_coupled_oscillator_rotation'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    buy_pr = day_ticks['BuyPr'].values
    sell_pr = day_ticks['SellPr'].values
    deal_pr = day_ticks['DealPrice'].values
    if len(deal_pr) < 20:
        return 0.0
    
    # 三個振盪器的速度
    d_buy = np.diff(buy_pr)
    d_sell = np.diff(sell_pr)
    d_deal = np.diff(deal_pr)
    
    n = len(d_buy)
    if n < 5:
        return 0.0
    
    # 標準化
    scale = max(np.std(d_deal) + 1e-10, 1e-6)
    d_buy_n = d_buy / scale
    d_sell_n = d_sell / scale
    d_deal_n = d_deal / scale
    
    # 3D 外積 (相鄰時刻)
    # v_t = (d_buy[t], d_sell[t], d_deal[t])
    # cross = v_{t-1} × v_t
    cx = d_sell_n[:-1] * d_deal_n[1:] - d_deal_n[:-1] * d_sell_n[1:]
    cy = d_deal_n[:-1] * d_buy_n[1:] - d_buy_n[:-1] * d_deal_n[1:]
    cz = d_buy_n[:-1] * d_sell_n[1:] - d_sell_n[:-1] * d_buy_n[1:]
    
    # 外積 Z 分量的均值 (捕捉旋轉方向)
    mean_cz = np.mean(cz)
    
    return mean_cz if np.isfinite(mean_cz) else 0.0


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
