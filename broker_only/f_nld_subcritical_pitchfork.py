"""
Feature: f_nld_subcritical_pitchfork
Top5 broker net flow 2nd derivative x asymmetric sign
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_subcritical_pitchfork'


def compute_feature(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock_id = df_stock['StockId'].iloc[0]
    dates = sorted(df_stock['Date'].unique())
    if len(dates) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    net_flows = []
    daily_asym = []
    for date in dates:
        day = df_stock[df_stock['Date'] == date]
        if len(day) < 3:
            net_flows.append(0.0)
            daily_asym.append(0)
            continue
        broker_agg = day.groupby('BrokerId').agg(
            buy_qty=('BuyQtm', 'sum'),
            sell_qty=('SellQtm', 'sum')
        ).reset_index()
        broker_agg['net'] = broker_agg['buy_qty'] - broker_agg['sell_qty']
        sorted_b = broker_agg.sort_values('net', ascending=False)
        net_flows.append(sorted_b.head(5)['net'].sum())
        daily_asym.append(np.sign(broker_agg['net'].sum()))
    flow_series = pd.Series(net_flows)
    d2 = safe_second_derivative(flow_series, ewm_span=3)
    result_df = pd.DataFrame({
        'StockId': stock_id, 'Date': dates,
        'raw': (d2 * pd.Series(daily_asym)).values
    })
    result_df['raw'] = result_df['raw'].fillna(0)
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    result_df[FEATURE_NAME] = rolling_zscore(result_df['raw'], 20)
    result_df[FEATURE_NAME] = safe_clip_fillna(result_df[FEATURE_NAME])
    return result_df[['StockId', 'Date', FEATURE_NAME]]
