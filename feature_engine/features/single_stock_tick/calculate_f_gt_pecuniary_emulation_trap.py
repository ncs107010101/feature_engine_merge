"""
Feature: f_gt_pecuniary_emulation_trap
Module: Game Theory v17 - Module 4
Theory: Pecuniary Emulation - Small trades mimic yesterday's large trades, but large have reversed
Direction: Positive → Retail mimics lagged signal while smart money reverses → Extreme LOW return
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
class FeatureGtPecuniaryEmulationTrap(BaseFeature):
    name = "f_gt_pecuniary_emulation_trap"
    description = "Game Theory Module 4: Pecuniary Emulation - Small trades mimic yesterday's large trades, but large have reversed. Retail mimics lagged signal while smart money reverses signals extreme LOW return."
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            raise ValueError("_tick_raw is required for f_gt_pecuniary_emulation_trap")
        stock_ids = data['StockId'].unique()
        results = []
        for stock_id in stock_ids:
            tick = tick_raw[tick_raw['StockId'] == stock_id].copy()
            if tick.empty:
                continue
            tick['Date'] = tick['Date'].astype(int)
            tick = tick.sort_values(['Date', 'TotalQty']).reset_index(drop=True)
            stock_result = _compute_single_stock(stock_id, tick)
            if not stock_result.empty:
                results.append(stock_result)
        if not results:
            return pd.DataFrame(columns=['StockId', 'Date', self.name])
        return pd.concat(results, ignore_index=True)


def _compute_single_stock(stock_id: str, tick: pd.DataFrame) -> pd.DataFrame:
    """Compute feature for a single stock."""
    if tick.empty:
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_pecuniary_emulation_trap'])
    
    FEATURE_NAME = 'f_gt_pecuniary_emulation_trap'
    
    daily_stats = tick.groupby('Date')['DealCount'].agg(
        q95=lambda x: x.quantile(0.95),
        q50=lambda x: x.quantile(0.50)
    ).reset_index()
    daily_stats = daily_stats.sort_values('Date').reset_index(drop=True)
    daily_stats['large_thresh'] = daily_stats['q95'].rolling(20, min_periods=1).mean().shift(1)
    daily_stats['small_thresh'] = daily_stats['q50'].rolling(20, min_periods=1).mean().shift(1)
    thresh_map = daily_stats.set_index('Date')[['large_thresh', 'small_thresh']].to_dict('index')
    
    results = []
    for date, day_ticks in tick.groupby('Date'):
        th = thresh_map.get(date)
        if th is None or pd.isna(th['large_thresh']) or pd.isna(th['small_thresh']):
            results.append({'Date': date, 'Small_NetFlow': 0.0, 'Large_NetFlow': 0.0})
            continue
        
        lt = th['large_thresh']
        st = th['small_thresh']
        dc = day_ticks['DealCount'].values
        pf = day_ticks['PrFlag'].values
        
        is_small = dc <= st
        is_large = dc >= lt
        
        small_buy = dc[is_small & (pf == 1)].sum()
        small_sell = dc[is_small & (pf == 0)].sum()
        large_buy = dc[is_large & (pf == 1)].sum()
        large_sell = dc[is_large & (pf == 0)].sum()
        
        results.append({
            'Date': date,
            'Small_NetFlow': float(small_buy - small_sell),
            'Large_NetFlow': float(large_buy - large_sell)
        })
    
    daily = pd.DataFrame(results).sort_values('Date').reset_index(drop=True)
    if daily.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    
    small_sign = np.sign(daily['Small_NetFlow'])
    large_sign = np.sign(daily['Large_NetFlow'])
    large_sign_lag = large_sign.shift(1)
    
    is_mimic = (small_sign == large_sign_lag).astype(int)
    is_reversal = (large_sign != large_sign_lag).astype(int)
    
    raw = is_mimic * is_reversal * np.abs(daily['Large_NetFlow'])
    out = _zscore_rolling(raw, 20)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    
    return pd.DataFrame({
        'StockId': stock_id,
        'Date': daily['Date'],
        FEATURE_NAME: out.values
    })