"""
Feature: f_gt_gradual_fleeing_capitulation
Module: Game Theory v17 - Module 1
Theory: Gradual Fleeing - Sequential declining active buy followed by extreme sell capitulation
Direction: Positive → Smart money abandons → Extreme LOW return
Data: hf_tick (trade_level1_data)
"""

import pandas as pd
import numpy as np

from ...core import BaseFeature, register_feature

EPS = 1e-8


def _zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / (rolling_std + EPS)."""
    rm = series.rolling(window, min_periods=max(1, window // 2)).mean()
    rs = series.rolling(window, min_periods=max(1, window // 2)).std()
    return (series - rm) / (rs + EPS)


@register_feature
class FeatureGtGradualFleeingCapitulation(BaseFeature):
    name = "f_gt_gradual_fleeing_capitulation"
    description = "Game Theory Module 1: Gradual Fleeing - Sequential declining active buy followed by extreme sell capitulation. Smart money abandons signal extreme LOW return."
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_data = kwargs.get('_tick_raw')
        if tick_data is None:
            raise ValueError("f_gt_gradual_fleeing_capitulation requires _tick_raw data from kwargs")
        stock_ids = data['StockId'].unique()
        results = []
        for stock_id in stock_ids:
            stock_tick = tick_data[tick_data['StockId'] == stock_id].copy()
            stock_result = _compute_single_stock(stock_id, stock_tick)
            if not stock_result.empty:
                results.append(stock_result)
        if not results:
            return pd.DataFrame(columns=['StockId', 'Date', self.name])
        return pd.concat(results, ignore_index=True)


def _compute_single_stock(stock_id: str, tick: pd.DataFrame) -> pd.DataFrame:
    """Compute feature for a single stock."""
    if tick.empty:
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_gradual_fleeing_capitulation'])
    
    FEATURE_NAME = 'f_gt_gradual_fleeing_capitulation'
    BIN_SIZE = 500
    
    # Assign bin index within each day
    tick['day_seq'] = tick.groupby('Date').cumcount()
    tick['bin_id'] = tick['day_seq'] // BIN_SIZE
    
    # Per-bin aggregation
    bins = tick.groupby(['Date', 'bin_id']).agg(
        active_buy_vol=('DealCount', lambda x: x[tick.loc[x.index, 'PrFlag'] == 1].sum()),
        active_sell_vol=('DealCount', lambda x: x[tick.loc[x.index, 'PrFlag'] == 0].sum()),
    ).reset_index()
    
    # Within each day, check consecutive declining active buy
    results = []
    for date, day_bins in bins.groupby('Date'):
        day_bins = day_bins.sort_values('bin_id').reset_index(drop=True)
        if len(day_bins) < 5:
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        abv = day_bins['active_buy_vol'].values
        asv = day_bins['active_sell_vol'].values
        
        # Consecutive declining active buy (4 bins)
        abv_diff = np.diff(abv)
        is_declining = (abv_diff < 0).astype(int)
        
        # Rolling sum of declining flags over 4 consecutive bins
        max_signal = 0.0
        sell_ma = pd.Series(asv).rolling(20, min_periods=1).mean().values
        for i in range(4, len(day_bins)):
            fleeing = int(np.sum(is_declining[i-4:i]) == 4)
            if fleeing and sell_ma[i] > 0:
                sell_surge = asv[i] / (sell_ma[i] + EPS)
                signal = sell_surge
                if signal > max_signal:
                    max_signal = signal
        results.append({'Date': date, 'raw': max_signal})
    
    daily = pd.DataFrame(results)
    if daily.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    
    daily = daily.sort_values('Date').reset_index(drop=True)
    raw_ewm = daily['raw'].ewm(span=3, min_periods=1).mean()
    out = _zscore_rolling(raw_ewm, 20)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    
    result = pd.DataFrame({
        'StockId': stock_id,
        'Date': daily['Date'],
        FEATURE_NAME: out.values
    })
    return result