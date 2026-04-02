"""
Feature: f_nld_symmetry_restoring_eigen
Institutional/retail flow transition matrix eigenvalue degeneracy rate
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (build_transition_matrix, leading_eigenvalue_real,
                          safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_symmetry_restoring_eigen'


def compute_feature(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock_id = df_stock['StockId'].iloc[0]
    dates = sorted(df_stock['Date'].unique())
    if len(dates) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    daily_eigenvalues = []
    daily_dates = []
    for date in dates:
        day = df_stock[df_stock['Date'] == date]
        if len(day) < 3:
            daily_eigenvalues.append(0.0)
            daily_dates.append(date)
            continue
        broker_agg = day.groupby('BrokerId').agg(
            buy_qty=('BuyQtm', 'sum'),
            sell_qty=('SellQtm', 'sum')
        ).reset_index()
        broker_agg['net'] = broker_agg['buy_qty'] - broker_agg['sell_qty']
        broker_agg['total'] = broker_agg['buy_qty'] + broker_agg['sell_qty']
        sorted_b = broker_agg.sort_values('total', ascending=False)
        inst = sorted_b.head(5)
        retail = sorted_b.iloc[5:] if len(sorted_b) > 5 else pd.DataFrame({'net': [0]})
        inst_in = max(inst['net'].clip(lower=0).sum(), 0) + 1
        inst_out = max((-inst['net'].clip(upper=0)).sum(), 0) + 1
        ret_in = max(retail['net'].clip(lower=0).sum(), 0) + 1
        ret_out = max((-retail['net'].clip(upper=0)).sum(), 0) + 1
        total = inst_in + inst_out + ret_in + ret_out
        matrix = np.array([
            [inst_in / total, inst_out / total],
            [ret_in / total, ret_out / total]
        ])
        ev = leading_eigenvalue_real(matrix)
        daily_eigenvalues.append(ev)
        daily_dates.append(date)
    result_df = pd.DataFrame({'StockId': stock_id, 'Date': daily_dates, 'raw': daily_eigenvalues})
    ev_series = pd.Series(daily_eigenvalues)
    d2 = safe_second_derivative(ev_series, ewm_span=3)
    final_day = df_stock[df_stock['Date'] == dates[-1]]
    if len(final_day) > 5:
        broker_agg = final_day.groupby('BrokerId').agg(net=('BuyQtm', 'sum')).reset_index()
        sorted_b = broker_agg.sort_values('net', ascending=False)
        retail_net = sorted_b.iloc[5:]['net'].sum() if len(sorted_b) > 5 else 0
        asym = np.sign(retail_net)
    else:
        asym = 0
    result_df['raw'] = (d2 * asym).values
    result_df['raw'] = result_df['raw'].fillna(0)
    result_df[FEATURE_NAME] = rolling_zscore(result_df['raw'], 20)
    result_df[FEATURE_NAME] = safe_clip_fillna(result_df[FEATURE_NAME])
    return result_df[['StockId', 'Date', FEATURE_NAME]]
