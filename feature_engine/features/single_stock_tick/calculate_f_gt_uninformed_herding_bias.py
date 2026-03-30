"""
Feature: f_gt_uninformed_herding_bias
Module: Game Theory v17 - Module 4
Theory: Uninformed Herding - Small trades blindly follow previous price direction
Direction: Positive → Blind herding dominates → Extreme LOW return
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
class FeatureGtUninformedHerdingBias(BaseFeature):
    name = "f_gt_uninformed_herding_bias"
    description = "Game Theory Module 4: Uninformed Herding - Small trades blindly follow previous price direction. Blind herding dominates signals extreme LOW return."
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            raise ValueError("_tick_raw is required for f_gt_uninformed_herding_bias")
        stock_ids = data['StockId'].unique()
        results = []
        for stock_id in stock_ids:
            tick = tick_raw[tick_raw['StockId'] == stock_id].copy()
            stock_result = _compute_single_stock(stock_id, tick)
            if not stock_result.empty:
                results.append(stock_result)
        if not results:
            return pd.DataFrame(columns=['StockId', 'Date', self.name])
        return pd.concat(results, ignore_index=True)


def _compute_single_stock(stock_id: str, tick: pd.DataFrame) -> pd.DataFrame:
    """Compute feature for a single stock."""
    if tick.empty:
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_uninformed_herding_bias'])
    
    FEATURE_NAME = 'f_gt_uninformed_herding_bias'
    BIN_SIZE = 100
    
    daily_stats = tick.groupby('Date')['DealCount'].agg(
        q50=lambda x: x.quantile(0.50)
    ).reset_index()
    daily_stats = daily_stats.sort_values('Date').reset_index(drop=True)
    daily_stats['small_thresh'] = daily_stats['q50'].rolling(20, min_periods=1).mean().shift(1)
    thresh_map = daily_stats.set_index('Date')['small_thresh'].to_dict()
    
    results = []
    for date, day_ticks in tick.groupby('Date'):
        st = thresh_map.get(date, np.nan)
        if pd.isna(st):
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        day_ticks = day_ticks.sort_values('TotalQty').reset_index(drop=True)
        n = len(day_ticks)
        if n < BIN_SIZE * 2:
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        dc = day_ticks['DealCount'].values
        dp = day_ticks['DealPrice'].values
        pf = day_ticks['PrFlag'].values
        
        price_dir = np.sign(np.diff(dp, prepend=dp[0]))
        trade_dir = pf * 2 - 1
        
        is_small = dc <= st
        
        prev_price_dir = np.roll(price_dir, 1)
        prev_price_dir[0] = 0
        
        follow = (trade_dir == prev_price_dir) & is_small & (prev_price_dir != 0)
        
        bin_ids = np.arange(n) // BIN_SIZE
        n_bins = bin_ids[-1] + 1
        
        follow_rates = []
        for b in range(n_bins):
            mask = bin_ids == b
            small_in_bin = np.sum(is_small[mask] & (prev_price_dir[mask] != 0))
            if small_in_bin > 0:
                follow_rates.append(np.sum(follow[mask]) / small_in_bin)
        
        daily_raw = np.mean(follow_rates) if follow_rates else 0.0
        results.append({'Date': date, 'raw': daily_raw})
    
    daily = pd.DataFrame(results).sort_values('Date').reset_index(drop=True)
    if daily.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    
    out = _zscore_rolling(daily['raw'], 20)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    
    return pd.DataFrame({
        'StockId': stock_id,
        'Date': daily['Date'],
        FEATURE_NAME: out.values
    })