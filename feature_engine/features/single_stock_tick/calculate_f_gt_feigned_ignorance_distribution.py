"""
Feature: f_gt_feigned_ignorance_distribution
Module: Game Theory v17 - Module 2
Theory: Feigning Ignorance (reverse) - Small active buys pump price, large passive sells distribute
Direction: Positive → Classic pump-and-dump → Extreme LOW return
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
class FeatureGtFeignedIgnoranceDistribution(BaseFeature):
    name = "f_gt_feigned_ignorance_distribution"
    description = "Game Theory Module 2: Feigning Ignorance (reverse) - Small active buys pump price, large passive sells distribute. Classic pump-and-dump signals extreme LOW return."
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            raise ValueError("_tick_raw is required for f_gt_feigned_ignorance_distribution")
        
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
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_feigned_ignorance_distribution'])
    
    FEATURE_NAME = 'f_gt_feigned_ignorance_distribution'
    BIN_SIZE = 200
    
    # Thresholds
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
        if date not in thresh_map:
            results.append({'Date': date, 'raw': 0.0})
            continue
        th = thresh_map[date]
        lt, st = th['large_thresh'], th['small_thresh']
        if pd.isna(lt) or pd.isna(st):
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        day_ticks = day_ticks.sort_values('TotalQty').reset_index(drop=True)
        n = len(day_ticks)
        if n < BIN_SIZE:
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        dc = day_ticks['DealCount'].values
        pf = day_ticks['PrFlag'].values
        
        # Outer market (PrFlag==1): small active buy pumps, large passive sell distributes
        outer = (pf == 1)
        small_active_buy = outer & (dc <= st)
        large_passive_sell = outer & (dc >= lt)
        
        bin_ids = np.arange(n) // BIN_SIZE
        n_bins = bin_ids[-1] + 1
        
        sab_count = np.zeros(n_bins)
        lps_vol = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_ids == b
            sab_count[b] = small_active_buy[mask].sum()
            lps_vol[b] = dc[mask & large_passive_sell].sum()
        
        raw_bins = sab_count * lps_vol
        daily_raw = np.mean(raw_bins) if len(raw_bins) > 0 else 0.0
        results.append({'Date': date, 'raw': daily_raw})
    
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