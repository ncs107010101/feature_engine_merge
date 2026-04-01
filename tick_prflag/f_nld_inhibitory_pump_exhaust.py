"""
特徵名: f_nld_inhibitory_pump_exhaust
靈感來源: Na+/K+ 幫浦耗竭 — 抑制性力量的二階崩潰
計算邏輯: 反向抵抗量(上漲時PrFlag=0量) → d² → min(0,d²) × sign(ΔP)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_inhibitory_pump_exhaust'


def _compute_day(day_ticks: pd.DataFrame) -> float:
    prices = day_ticks['DealPrice'].values
    volumes = day_ticks['DealCount'].values.astype(float)
    prflag = day_ticks['PrFlag'].values
    if len(prices) < 30:
        return 0.0
    
    dp = np.diff(prices)
    # 上漲期間的內盤成交量 (抵抗力)
    # 下跌期間的外盤成交量 (也是抵抗力)
    resist = np.zeros(len(dp))
    for i in range(len(dp)):
        if dp[i] > 0 and prflag[i+1] == 0:
            resist[i] = volumes[i+1]  # 上漲時賣壓
        elif dp[i] < 0 and prflag[i+1] == 1:
            resist[i] = volumes[i+1]  # 下跌時買壓
    
    resist_series = pd.Series(resist)
    # EWM 平滑
    resist_smooth = resist_series.ewm(span=20, min_periods=1).mean()
    
    # 二階導數
    d2 = safe_second_derivative(resist_smooth, ewm_span=10)
    
    # 取負加速度部分 (抵抗力在枯竭)
    d2_neg = d2.clip(upper=0)
    mean_d2_neg = d2_neg.mean()
    
    if not np.isfinite(mean_d2_neg):
        return 0.0
    
    asym = np.sign(prices[-1] - prices[0])
    return mean_d2_neg * asym


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
