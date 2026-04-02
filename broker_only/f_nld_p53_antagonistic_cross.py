"""
Feature: f_nld_p53_antagonistic_cross
Brake/killer broker flow rotation via 2D cross product
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (cross_product_2d, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_p53_antagonistic_cross'


def compute_feature(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock_id = df_stock['StockId'].iloc[0]
    dates = sorted(df_stock['Date'].unique())
    if len(dates) < 3:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    brake_flows = []
    killer_flows = []
    for date in dates:
        day = df_stock[df_stock['Date'] == date]
        if len(day) < 3:
            brake_flows.append(0.0)
            killer_flows.append(0.0)
            continue
        broker_agg = day.groupby('BrokerId').agg(
            buy_qty=('BuyQtm', 'sum'),
            sell_qty=('SellQtm', 'sum')
        ).reset_index()
        broker_agg['net'] = broker_agg['buy_qty'] - broker_agg['sell_qty']
        sorted_b = broker_agg.sort_values('net', ascending=False)
        brake_flows.append(sorted_b.head(3)['net'].sum())
        killer_flows.append(sorted_b.tail(3)['net'].sum())
    brake = np.array(brake_flows, dtype=float)
    killer = np.array(killer_flows, dtype=float)
    b_std = np.std(brake) + 1e-10
    k_std = np.std(killer) + 1e-10
    cross = cross_product_2d(
        brake[:-1] / b_std, killer[:-1] / k_std,
        brake[1:] / b_std, killer[1:] / k_std
    )
    result_df = pd.DataFrame({'StockId': stock_id, 'Date': dates[1:], 'raw': cross})
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    result_df[FEATURE_NAME] = rolling_zscore(result_df['raw'], 20)
    result_df[FEATURE_NAME] = safe_clip_fillna(result_df[FEATURE_NAME])
    return result_df[['StockId', 'Date', FEATURE_NAME]]
