"""
Feature: f_gt_forward_induction_signal
Module: Game Theory v17 - Module 1
Theory: Forward Induction - Major players quiet in AM, aggressive large buys in PM
Direction: Positive → Stealth accumulation via delayed action → Extreme HIGH return
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
class FeatureGtForwardInductionSignal(BaseFeature):
    name = "f_gt_forward_induction_signal"
    description = "Game Theory Module 1: Forward Induction - Major players quiet in AM, aggressive large buys in PM. Stealth accumulation via delayed action signals extreme HIGH return."
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_data = kwargs.get('_tick_raw')
        if tick_data is None:
            raise ValueError("f_gt_forward_induction_signal requires _tick_raw data from kwargs")
        
        stock_ids = data['StockId'].unique()
        results = []
        for stock_id in stock_ids:
            tick = tick_data[tick_data['StockId'] == stock_id].copy()
            stock_result = _compute_single_stock(stock_id, tick)
            if not stock_result.empty:
                results.append(stock_result)
        if not results:
            return pd.DataFrame(columns=['StockId', 'Date', self.name])
        return pd.concat(results, ignore_index=True)


def _compute_single_stock(stock_id: str, tick: pd.DataFrame) -> pd.DataFrame:
    """Compute feature for a single stock."""
    if tick.empty:
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_forward_induction_signal'])
    
    FEATURE_NAME = 'f_gt_forward_induction_signal'
    HEAD_TICKS = 1000
    TAIL_TICKS = 1000
    
    daily_q95 = tick.groupby('Date')['DealCount'].quantile(0.95).reset_index()
    daily_q95.columns = ['Date', 'daily_q95']
    daily_q95 = daily_q95.sort_values('Date').reset_index(drop=True)
    daily_q95['large_thresh'] = daily_q95['daily_q95'].rolling(20, min_periods=1).mean().shift(1)
    
    results = []
    am_large_buy_list = []
    
    for date, day_ticks in tick.groupby('Date'):
        thresh_row = daily_q95[daily_q95['Date'] == date]
        if thresh_row.empty or pd.isna(thresh_row['large_thresh'].values[0]):
            results.append({'Date': date, 'AM_Large_Buy': 0.0, 'PM_Large_Buy': 0.0})
            am_large_buy_list.append(0.0)
            continue
        
        large_thresh = thresh_row['large_thresh'].values[0]
        day_ticks = day_ticks.sort_values('TotalQty').reset_index(drop=True)
        n = len(day_ticks)
        
        am = day_ticks.head(min(HEAD_TICKS, n))
        am_mask = (am['PrFlag'] == 1) & (am['DealCount'] >= large_thresh)
        am_large_buy = am.loc[am_mask, 'DealCount'].sum()
        
        pm = day_ticks.tail(min(TAIL_TICKS, n))
        pm_mask = (pm['PrFlag'] == 1) & (pm['DealCount'] >= large_thresh)
        pm_large_buy = pm.loc[pm_mask, 'DealCount'].sum()
        
        results.append({'Date': date, 'AM_Large_Buy': float(am_large_buy), 'PM_Large_Buy': float(pm_large_buy)})
        am_large_buy_list.append(float(am_large_buy))
    
    daily = pd.DataFrame(results)
    if daily.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    
    daily = daily.sort_values('Date').reset_index(drop=True)
    
    am_series = pd.Series(am_large_buy_list)
    am_mean = am_series.rolling(20, min_periods=1).mean().shift(1).fillna(0)
    is_am_quiet = (daily['AM_Large_Buy'] < am_mean.values).astype(int)
    
    ratio = daily['PM_Large_Buy'] / (daily['AM_Large_Buy'] + EPS)
    raw = ratio * is_am_quiet
    out = _zscore_rolling(raw, 20)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    
    result = pd.DataFrame({
        'StockId': stock_id,
        'Date': daily['Date'],
        FEATURE_NAME: out.values
    })
    return result