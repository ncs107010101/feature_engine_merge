"""
Feature: f_nld_fvs_controlling_dot
FVS controlling node acceleration dot product
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_utils import (safe_second_derivative, rolling_zscore, safe_clip_fillna)

FEATURE_NAME = 'f_nld_fvs_controlling_dot'


def compute_feature(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock_id = df_stock['StockId'].iloc[0]
    dates = sorted(df_stock['Date'].unique())
    if len(dates) < 5:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    ctrl_flows = []
    retail_flows = []
    for date in dates:
        day = df_stock[df_stock['Date'] == date]
        if len(day) < 3:
            ctrl_flows.append(0.0)
            retail_flows.append(0.0)
            continue
        broker_agg = day.groupby('BrokerId').agg(
            buy_qty=('BuyQtm', 'sum'),
            sell_qty=('SellQtm', 'sum')
        ).reset_index()
        broker_agg['net'] = broker_agg['buy_qty'] - broker_agg['sell_qty']
        broker_agg['total_vol'] = broker_agg['buy_qty'] + broker_agg['sell_qty']
        
        sorted_b = broker_agg.sort_values('total_vol', ascending=False)
        ctrl = sorted_b.head(5)
        retail = sorted_b.iloc[5:] if len(sorted_b) > 5 else pd.DataFrame({'net': [0]})
        ctrl_flows.append(ctrl['net'].sum())
        retail_flows.append(retail['net'].sum())
    ctrl_series = pd.Series(ctrl_flows)
    retail_series = pd.Series(retail_flows)
    d2_ctrl = safe_second_derivative(ctrl_series, ewm_span=3)
    d2_retail = safe_second_derivative(retail_series, ewm_span=3)
    dot = d2_ctrl * d2_retail
    fvs_sign = np.sign(ctrl_series)
    raw = (dot * fvs_sign).fillna(0).values
    result_df = pd.DataFrame({'StockId': stock_id, 'Date': dates, 'raw': raw})
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    result_df[FEATURE_NAME] = rolling_zscore(result_df['raw'], 20)
    result_df[FEATURE_NAME] = safe_clip_fillna(result_df[FEATURE_NAME])
    return result_df[['StockId', 'Date', FEATURE_NAME]]
