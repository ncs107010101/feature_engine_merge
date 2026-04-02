"""
Feature: f_nld_hopf_partial_antiphase
3-agent (buy-side/sell-side/others) 3D cross product Z-component
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_hopf_partial_antiphase'


def compute_feature(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock_id = df_stock['StockId'].iloc[0]
    dates = sorted(df_stock['Date'].unique())
    if len(dates) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    buyside = []
    sellside = []
    others = []
    for date in dates:
        day = df_stock[df_stock['Date'] == date]
        if len(day) < 3:
            buyside.append(0.0); sellside.append(0.0); others.append(0.0)
            continue
        broker_agg = day.groupby('BrokerId').agg(
            buy_qty=('BuyQtm', 'sum'), sell_qty=('SellQtm', 'sum')
        ).reset_index()
        broker_agg['net'] = broker_agg['buy_qty'] - broker_agg['sell_qty']
        sorted_b = broker_agg.sort_values('net', ascending=False)
        buyside.append(sorted_b.head(5)['net'].sum())
        sellside.append(sorted_b.tail(5)['net'].sum())
        rest = sorted_b.iloc[5:-5]['net'].sum() if len(sorted_b) > 10 else 0
        others.append(rest)
    buyside = np.array(buyside, dtype=float)
    sellside = np.array(sellside, dtype=float)
    others = np.array(others, dtype=float)
    d_buy = np.diff(buyside)
    d_sell = np.diff(sellside)
    d_other = np.diff(others)
    if len(d_buy) < 3:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    scale = max(np.std(d_buy), np.std(d_sell), np.std(d_other), 1e-10)
    cz = (d_buy / scale) * (d_sell / scale) - (d_sell / scale) * (d_other / scale)
    result_df = pd.DataFrame({'StockId': stock_id, 'Date': dates[1:], 'raw': cz})
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    result_df[FEATURE_NAME] = rolling_zscore(result_df['raw'], 20)
    result_df[FEATURE_NAME] = safe_clip_fillna(result_df[FEATURE_NAME])
    return result_df[['StockId', 'Date', FEATURE_NAME]]
