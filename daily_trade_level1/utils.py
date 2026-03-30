"""
共用工具函數 - Daily + Tick 跨頻率特徵
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'G:\AI_dataAnalysis\data'
TICK_DIR = os.path.join(DATA_DIR, 'trade_level1_data')
DAILY_DIR = os.path.join(DATA_DIR, 'daily_data_bystock')
FEATURE_DF_DIR = r'G:\project_extreme_feature\20260328_v1\new_feature_create\feature_df'
os.makedirs(FEATURE_DF_DIR, exist_ok=True)


def load_tick_data(stock_id, date_start=20200101, date_end=20251231):
    fpath = os.path.join(TICK_DIR, f'{stock_id}.parquet')
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df['Date_int'] = df['Date'].astype(int)
    df = df[(df['Date_int'] >= date_start) & (df['Date_int'] <= date_end)].copy()
    df = df.sort_values(['Date_int', 'TotalQty']).reset_index(drop=True)
    return df


def load_daily_data(stock_id, date_start=20200101, date_end=20251231):
    fpath = os.path.join(DAILY_DIR, f'{stock_id}.parquet')
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath, engine='pyarrow')
    df = df[(df['Date'] >= date_start) & (df['Date'] <= date_end)].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def compute_tick_daily_intermediates(df_tick):
    """計算 tick 日級中間量"""
    if df_tick.empty:
        return pd.DataFrame()

    records = []
    for date_int, grp in df_tick.groupby('Date_int'):
        dp = grp['DealPrice'].values.astype(np.float64)
        dc = grp['DealCount'].values.astype(np.float64)
        pf = grp['PrFlag'].values
        n = len(dp)

        if n < 5:
            continue

        # VWAP
        vwap = np.sum(dp * dc) / (np.sum(dc) + 1e-10)
        # 價格斜率
        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, dp, 1)[0] if n > 2 else 0.0
        # 收開
        open_p, close_p = dp[0], dp[-1]
        # 大小單
        q85 = np.percentile(dc, 85) if n > 10 else dc.max()
        buy_vol = np.sum(dc[pf == 1])
        sell_vol = np.sum(dc[pf == 0])
        large_buy = np.sum(dc[(dc >= q85) & (pf == 1)])
        small_buy = np.sum(dc[(dc < q85) & (pf == 1)])
        total_vol = np.sum(dc)
        # 高於均價的買量
        above_vwap_buy = np.sum(dc[(dp > vwap * 0.98) & (pf == 1)])
        above_vwap_high_buy = np.sum(dc[(dp > vwap * 1.02) & (pf == 1)])
        # 低於均價的買量(冷區)
        cold_zone_buy = np.sum(dc[(dp < vwap) & (pf == 1)])
        # 冷區斜率
        cold_mask = dp < vwap
        cold_prices = dp[cold_mask]
        cold_slope = np.polyfit(np.arange(len(cold_prices)), cold_prices, 1)[0] if len(cold_prices) > 5 else 0.0
        # tick 均價
        tick_mean_price = np.mean(dp)

        records.append({
            'Date': date_int,
            'vwap': vwap,
            'tick_slope': slope,
            'open_p': open_p,
            'close_p': close_p,
            'buy_vol': buy_vol,
            'sell_vol': sell_vol,
            'large_buy': large_buy,
            'small_buy': small_buy,
            'total_vol': total_vol,
            'above_vwap_buy': above_vwap_buy,
            'above_vwap_high_buy': above_vwap_high_buy,
            'cold_zone_buy': cold_zone_buy,
            'cold_slope': cold_slope,
            'tick_mean_price': tick_mean_price,
        })

    return pd.DataFrame(records)


def rolling_zscore(series, window=20, min_periods=None):
    if min_periods is None:
        min_periods = max(1, window // 2)
    mu = series.rolling(window, min_periods=min_periods).mean()
    sigma = series.rolling(window, min_periods=min_periods).std()
    return (series - mu) / (sigma + 1e-10)


def post_process(series, clip_val=5):
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip_val, clip_val)


def save_feature(df_feature, feature_name):
    fpath = os.path.join(FEATURE_DF_DIR, f'{feature_name}.parquet')
    df_feature.to_parquet(fpath, index=False, engine='pyarrow')
    print(f"Saved {feature_name} -> {fpath}, shape={df_feature.shape}")


def get_all_stock_ids():
    tick_ids = set(f.replace('.parquet', '') for f in os.listdir(TICK_DIR) if f.endswith('.parquet'))
    daily_ids = set(f.replace('.parquet', '') for f in os.listdir(DAILY_DIR) if f.endswith('.parquet'))
    return sorted(tick_ids & daily_ids)
