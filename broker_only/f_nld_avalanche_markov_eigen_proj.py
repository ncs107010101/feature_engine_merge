"""
Feature: f_nld_avalanche_markov_eigen_proj
4-state Markov transition matrix stationary distribution projection
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (build_transition_matrix, stationary_distribution,
                          rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_avalanche_markov_eigen_proj'


def compute_feature(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock_id = df_stock['StockId'].iloc[0]
    dates = sorted(df_stock['Date'].unique())
    if len(dates) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    daily_states = []
    for date in dates:
        day = df_stock[df_stock['Date'] == date]
        if len(day) < 3:
            daily_states.append(1)
            continue
        broker_agg = day.groupby('BrokerId').agg(
            buy_qty=('BuyQtm', 'sum'),
            sell_qty=('SellQtm', 'sum')
        ).reset_index()
        broker_agg['net'] = broker_agg['buy_qty'] - broker_agg['sell_qty']
        total_net = broker_agg['net'].sum()
        if total_net > 0:
            daily_states.append(3)
        elif total_net < 0:
            daily_states.append(0)
        else:
            daily_states.append(1)
    four_states = np.zeros(len(daily_states), dtype=int)
    for i in range(1, len(daily_states)):
        prev = daily_states[i-1]
        curr = daily_states[i]
        if prev <= 1 and curr <= 1:
            four_states[i] = 0
        elif prev <= 1 and curr >= 2:
            four_states[i] = 1
        elif prev >= 2 and curr <= 1:
            four_states[i] = 2
        else:
            four_states[i] = 3
    window = 20
    projection = np.array([-1.0, -0.5, 0.5, 1.0])
    proj_values = []
    for i in range(len(four_states)):
        start = max(0, i - window + 1)
        window_states = four_states[start:i+1]
        if len(window_states) < 5:
            proj_values.append(0.0)
            continue
        tm = build_transition_matrix(window_states, n_states=4, laplace_smooth=1.0)
        pi = stationary_distribution(tm)
        proj_values.append(np.dot(pi, projection))
    result_df = pd.DataFrame({'StockId': stock_id, 'Date': dates, 'raw': proj_values})
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    result_df[FEATURE_NAME] = rolling_zscore(result_df['raw'], 20)
    result_df[FEATURE_NAME] = safe_clip_fillna(result_df[FEATURE_NAME])
    return result_df[['StockId', 'Date', FEATURE_NAME]]
