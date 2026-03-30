"""
Feature: f_gt_persuasion_suspense_slide
Module: Game Theory v17 - Module 2
Theory: Suspense and Slide - Extended silence followed by aggressive large buy ambush
Direction: Positive → End of silence, major attack → Extreme HIGH return
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
class FeatureGtPersuasionSuspenseSlide(BaseFeature):
    name = "f_gt_persuasion_suspense_slide"
    description = "Game Theory Module 2: Suspense and Slide - Extended silence followed by aggressive large buy ambush. End of silence and major attack signals extreme HIGH return."
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            raise ValueError("_tick_raw is required for f_gt_persuasion_suspense_slide")
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
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_persuasion_suspense_slide'])
    
    FEATURE_NAME = 'f_gt_persuasion_suspense_slide'
    BIN_SIZE = 100
    
    daily_stats = tick.groupby('Date')['DealCount'].agg(
        q95=lambda x: x.quantile(0.95)
    ).reset_index()
    daily_stats = daily_stats.sort_values('Date').reset_index(drop=True)
    daily_stats['large_thresh'] = daily_stats['q95'].rolling(20, min_periods=1).mean().shift(1)
    thresh_map = daily_stats.set_index('Date')['large_thresh'].to_dict()
    
    results = []
    for date, day_ticks in tick.groupby('Date'):
        lt = thresh_map.get(date, np.nan)
        if pd.isna(lt):
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        day_ticks = day_ticks.sort_values('TotalQty').reset_index(drop=True)
        n = len(day_ticks)
        if n < BIN_SIZE * 6:
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        dc = day_ticks['DealCount'].values
        pf = day_ticks['PrFlag'].values
        
        bin_ids = np.arange(n) // BIN_SIZE
        n_bins = bin_ids[-1] + 1
        
        bin_vol = np.zeros(n_bins)
        large_buy_vol = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_ids == b
            bin_vol[b] = dc[mask].sum()
            lb_mask = mask & (pf == 1) & (dc >= lt)
            large_buy_vol[b] = dc[lb_mask].sum()
        
        vol_ma = pd.Series(bin_vol).rolling(20, min_periods=1).mean().shift(1).values
        silence_flag = np.zeros(n_bins)
        for b in range(1, n_bins):
            if not np.isnan(vol_ma[b]) and vol_ma[b] > 0:
                silence_flag[b] = 1.0 if bin_vol[b] < 0.5 * vol_ma[b] else 0.0
        
        silence_streak = pd.Series(silence_flag).rolling(5, min_periods=1).sum().shift(1).fillna(0).values
        
        signal = silence_streak * large_buy_vol
        max_signal = np.max(signal) if len(signal) > 0 else 0.0
        results.append({'Date': date, 'raw': max_signal})
    
    daily = pd.DataFrame(results)
    if daily.empty:
        return pd.DataFrame(columns=['StockId', 'Date', FEATURE_NAME])
    
    daily = daily.sort_values('Date').reset_index(drop=True)
    out = _zscore_rolling(daily['raw'], 20)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    
    result = pd.DataFrame({
        'StockId': stock_id,
        'Date': daily['Date'],
        FEATURE_NAME: out.values
    })
    return result