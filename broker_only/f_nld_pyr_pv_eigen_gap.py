"""
Feature: f_nld_pyr_pv_eigen_gap
Pyramidal-PV eigenvalue gap from top buyer/seller broker interaction matrix
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (eigenvalue_gap, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_pyr_pv_eigen_gap'


def _compute_day(day_broker: pd.DataFrame) -> float:
    if len(day_broker) < 5:
        return 0.0
    broker_agg = day_broker.groupby('BrokerId').agg(
        buy_qty=('BuyQtm', 'sum'),
        sell_qty=('SellQtm', 'sum')
    ).reset_index()
    broker_agg['net'] = broker_agg['buy_qty'] - broker_agg['sell_qty']
    if len(broker_agg) < 3:
        return 0.0
    sorted_brokers = broker_agg.sort_values('net', ascending=False)
    top5_buy = sorted_brokers.head(5)
    top5_sell = sorted_brokers.tail(5)
    bb = top5_buy['buy_qty'].sum()
    bs = top5_buy['sell_qty'].sum()
    sb = top5_sell['buy_qty'].sum()
    ss = top5_sell['sell_qty'].sum()
    total = bb + bs + sb + ss + 4
    matrix = np.array([
        [(bb + 1) / total, (bs + 1) / total],
        [(sb + 1) / total, (ss + 1) / total]
    ])
    gap = eigenvalue_gap(matrix)
    net_flow = (top5_buy['net'].sum() + top5_sell['net'].sum())
    asym = np.sign(net_flow) if net_flow != 0 else 0
    return gap * asym


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
