"""
共用工具函數 - 大氣流體力學特徵計算
"""
import numpy as np
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'G:\AI_dataAnalysis\data'
TICK_DIR = os.path.join(DATA_DIR, 'trade_level1_data')
BROKER_DIR = os.path.join(DATA_DIR, 'broker_data')
DAILY_DIR = os.path.join(DATA_DIR, 'daily_data_bystock')
FEATURE_DF_DIR = r'G:\project_extreme_feature\20260328_v1\new_feature_create\feature_df'

os.makedirs(FEATURE_DF_DIR, exist_ok=True)


def load_tick_data(stock_id, date_start=20200101, date_end=20251231):
    """載入單一股票的 tick 數據"""
    fpath = os.path.join(TICK_DIR, f'{stock_id}.parquet')
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    # Date is string in tick data, convert to int for filtering
    df['Date_int'] = df['Date'].astype(int)
    df = df[(df['Date_int'] >= date_start) & (df['Date_int'] <= date_end)].copy()
    # Sort by TotalQty (deterministic within same second)
    df = df.sort_values(['Date_int', 'TotalQty']).reset_index(drop=True)
    return df


def rolling_zscore(series, window=20, min_periods=None):
    """滾動 Z-score 標準化"""
    if min_periods is None:
        min_periods = max(1, window // 2)
    mu = series.rolling(window, min_periods=min_periods).mean()
    sigma = series.rolling(window, min_periods=min_periods).std()
    return ((series - mu) / (sigma + 1e-10))


def post_process(series, clip_val=5):
    """後處理：replace inf, clip, fillna"""
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip_val, clip_val)


def get_all_stock_ids():
    """獲取所有有 tick 資料的股票列表"""
    files = [f.replace('.parquet', '') for f in os.listdir(TICK_DIR) if f.endswith('.parquet')]
    return sorted(files)


def save_feature(df_feature, feature_name):
    """儲存特徵 DataFrame 為 parquet"""
    fpath = os.path.join(FEATURE_DF_DIR, f'{feature_name}.parquet')
    df_feature.to_parquet(fpath, index=False, engine='pyarrow')
    print(f"Saved {feature_name} -> {fpath}, shape={df_feature.shape}")


def preprocess_tick_for_day(df_day):
    """
    預處理單日 tick 數據，計算常用中間變數
    df_day: 單一股票單一天的 tick 數據，已排序
    """
    if len(df_day) < 5:
        return None

    dp = df_day['DealPrice'].values
    dc = df_day['DealCount'].values
    pf = df_day['PrFlag'].values
    bp = df_day['BuyPr'].values
    sp = df_day['SellPr'].values
    ts = df_day['DealTimeSecond'].values  
    tq = df_day['TotalQty'].values

    # 價格變動
    delta_p = np.diff(dp, prepend=dp[0])
    # 主動買=PrFlag 1, 主動賣=PrFlag 0
    buy_mask = (pf == 1)
    sell_mask = (pf == 0)
    # 買賣方向 sign: +1 for buy, -1 for sell, 0 for other
    direction = np.where(buy_mask, 1, np.where(sell_mask, -1, 0))
    # Spread
    spread = sp - bp

    return {
        'DealPrice': dp,
        'DealCount': dc,
        'PrFlag': pf,
        'BuyPr': bp,
        'SellPr': sp,
        'DealTimeSecond': ts,
        'TotalQty': tq,
        'delta_p': delta_p,
        'buy_mask': buy_mask,
        'sell_mask': sell_mask,
        'direction': direction,
        'spread': spread,
        'n': len(dp),
        'open_price': dp[0],
        'close_price': dp[-1],
    }
