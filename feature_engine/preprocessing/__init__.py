"""
feature_engine.preprocessing
================================
Router for preprocessing raw data based on its combination type.
"""

import pandas as pd
from typing import Union, Tuple

from .single_stock_daily import preprocess_single_stock_daily
from .single_stock_tick import preprocess_single_stock_tick
from .single_stock_broker import preprocess_single_stock_broker
from .cross_broker_tick import preprocess_cross_broker_tick
from .cross_broker_daily import preprocess_cross_broker_daily
from .cross_tick_daily import preprocess_cross_tick_daily
from .cross_tick_broker_daily import preprocess_cross_tick_broker_daily

def preprocess(data_combination: str, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    df = data.copy()
    return_broker_day = kwargs.pop('return_broker_day', False)
    
    if data_combination == "single_stock_daily":
        return preprocess_single_stock_daily(df)
    elif data_combination == "single_stock_tick":
        return preprocess_single_stock_tick(df)
    elif data_combination == "single_stock_broker":
        return preprocess_single_stock_broker(df, return_broker_day=return_broker_day)
    elif data_combination == "cross_broker_tick":
        return preprocess_cross_broker_tick(df, **kwargs)
    elif data_combination == "cross_broker_daily":
        kwargs.pop('return_broker_day', None)
        return preprocess_cross_broker_daily(df, return_broker_day=return_broker_day, **kwargs)
    elif data_combination == "cross_tick_daily":
        return preprocess_cross_tick_daily(df, **kwargs)
    elif data_combination == "cross_tick_broker_daily":
        kwargs.pop('return_broker_day', None)
        return preprocess_cross_tick_broker_daily(df, return_broker_day=return_broker_day, **kwargs)
    else:
        return df
