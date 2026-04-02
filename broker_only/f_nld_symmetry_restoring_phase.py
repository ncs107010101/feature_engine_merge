"""
Feature: f_nld_symmetry_restoring_phase
HHI phase-space angular acceleration
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (phase_angle, safe_second_derivative,
                          rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_symmetry_restoring_phase'


def _compute_hhi(series: pd.Series) -> float:
    total = series.sum()
    if total == 0:
        return 0
    shares = series / total
    return (shares ** 2).sum()


def compute_feature(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock_id = df_stock['StockId'].iloc[0]
    dates = sorted(df_stock['Date'].unique())
    if len(dates) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    hhi_buy_list = []
    hhi_sell_list = []
    asym_list = []
    for date in dates:
        day = df_stock[df_stock['Date'] == date]
        if len(day) < 3:
            hhi_buy_list.append(0.0)
            hhi_sell_list.append(0.0)
            asym_list.append(0)
            continue
        broker_agg = day.groupby('BrokerId').agg(
            buy_qty=('BuyQtm', 'sum'),
            sell_qty=('SellQtm', 'sum')
        ).reset_index()
        hhi_buy = _compute_hhi(broker_agg['buy_qty'])
        hhi_sell = _compute_hhi(broker_agg['sell_qty'])
        hhi_buy_list.append(hhi_buy)
        hhi_sell_list.append(hhi_sell)
        net = broker_agg['buy_qty'].sum() - broker_agg['sell_qty'].sum()
        asym_list.append(np.sign(net))
    hhi_buy_arr = np.array(hhi_buy_list)
    hhi_sell_arr = np.array(hhi_sell_list)
    theta = phase_angle(hhi_sell_arr + 1e-10, hhi_buy_arr + 1e-10)
    theta_series = pd.Series(theta)
    d2 = safe_second_derivative(theta_series, ewm_span=3)
    raw = (d2 * pd.Series(asym_list)).fillna(0).values
    result_df = pd.DataFrame({'StockId': stock_id, 'Date': dates, 'raw': raw})
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    result_df[FEATURE_NAME] = rolling_zscore(result_df['raw'], 20)
    result_df[FEATURE_NAME] = safe_clip_fillna(result_df[FEATURE_NAME])
    return result_df[['StockId', 'Date', FEATURE_NAME]]
