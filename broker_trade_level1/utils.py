"""
共用工具函數 - Broker + Tick 跨資料特徵
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'G:\AI_dataAnalysis\data'
TICK_DIR = os.path.join(DATA_DIR, 'trade_level1_data')
BROKER_DIR = os.path.join(DATA_DIR, 'broker_data')
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


def load_broker_data(stock_id, date_start=20200101, date_end=20251231):
    fpath = os.path.join(BROKER_DIR, f'{stock_id}.parquet')
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath)
    df = df[(df['Date'] >= date_start) & (df['Date'] <= date_end)].copy()
    df = df.sort_values(['Date', 'BrokerId']).reset_index(drop=True)
    return df


def load_daily_data(stock_id, date_start=20200101, date_end=20251231):
    fpath = os.path.join(DAILY_DIR, f'{stock_id}.parquet')
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_parquet(fpath, engine='pyarrow')
    df = df[(df['Date'] >= date_start) & (df['Date'] <= date_end)].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def preprocess_broker_daily(df_broker):
    """
    將 broker_data 聚合為日級別的券商特徵
    Returns DataFrame with: Date, Top10_NetBuy, Retail_NetBuy, NB_Brokers, etc.
    """
    if df_broker.empty:
        return pd.DataFrame()

    # 日級 per-broker 聚合
    broker_daily = df_broker.groupby(['Date', 'BrokerId']).agg(
        TotalBuy=('BuyQtm', 'sum'),
        TotalSell=('SellQtm', 'sum')
    ).reset_index()
    broker_daily['NetBuy'] = broker_daily['TotalBuy'] - broker_daily['TotalSell']

    result = []
    for date, grp in broker_daily.groupby('Date'):
        grp_sorted = grp.sort_values('TotalBuy', ascending=False)
        n_brokers = len(grp)

        # Top10 by total buy
        top10 = grp_sorted.head(min(10, n_brokers))
        rest = grp_sorted.iloc[min(10, n_brokers):]

        top10_net = top10['NetBuy'].sum()
        top5_net = grp_sorted.head(min(5, n_brokers))['NetBuy'].sum()
        retail_net = rest['NetBuy'].sum()
        total_buy = grp['TotalBuy'].sum()
        total_sell = grp['TotalSell'].sum()

        # 淨買超的不同券商數 (寬度)
        nb_buy_brokers = (grp['NetBuy'] > 0).sum()

        result.append({
            'Date': date,
            'Top10_NetBuy': top10_net,
            'Top5_NetBuy': top5_net,
            'Retail_NetBuy': retail_net,
            'TotalBuy': total_buy,
            'TotalSell': total_sell,
            'NB_BuyBrokers': nb_buy_brokers,
            'N_Brokers': n_brokers,
        })

    return pd.DataFrame(result)


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
    # 取 tick 和 broker 都有的股票
    tick_ids = set(f.replace('.parquet', '') for f in os.listdir(TICK_DIR) if f.endswith('.parquet'))
    broker_ids = set(f.replace('.parquet', '') for f in os.listdir(BROKER_DIR) if f.endswith('.parquet'))
    return sorted(tick_ids & broker_ids)
