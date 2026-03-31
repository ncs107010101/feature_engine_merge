"""
Feature: f_gt_cheap_talk_distraction_trap
Module: Game Theory v17 - Module 3
Theory: Cheap Talk Distraction - 1-lot noise buys mask real large-lot selling
Direction: Positive → Large sell per 1-lot noise → Extreme LOW return
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
class FeatureGtCheapTalkDistractionTrap(BaseFeature):
    name = "f_gt_cheap_talk_distraction_trap"
    description = "Game Theory Module 3: 1-lot noise buys mask real large-lot selling. Large sell volume per 1-lot buy count indicates hidden institutional distribution."
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            raise ValueError("_tick_raw is required for f_gt_cheap_talk_distraction_trap")
        
        # Get unique stock IDs from input data
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
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_cheap_talk_distraction_trap'])
    
    # Threshold: rolling 20-day mean of q95, shifted by 1 day
    daily_stats = tick.groupby('Date')['DealCount'].agg(
        q95=lambda x: x.quantile(0.95)
    ).reset_index()
    daily_stats = daily_stats.sort_values('Date').reset_index(drop=True)
    daily_stats['large_thresh'] = daily_stats['q95'].rolling(20, min_periods=1).mean().shift(1)
    thresh_map = daily_stats.set_index('Date')['large_thresh'].to_dict()
    
    BIN_SIZE = 100
    results = []
    
    for date, day_ticks in tick.groupby('Date'):
        lt = thresh_map.get(date, np.nan)
        if pd.isna(lt):
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        day_ticks = day_ticks.sort_values('TotalQty').reset_index(drop=True)
        n = len(day_ticks)
        if n < BIN_SIZE:
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        dc = day_ticks['DealCount'].values
        pf = day_ticks['PrFlag'].values
        
        bin_ids = np.arange(n) // BIN_SIZE
        n_bins = bin_ids[-1] + 1
        
        one_lot_buy_count = np.zeros(n_bins)
        large_sell_vol = np.zeros(n_bins)
        
        for b in range(n_bins):
            mask = bin_ids == b
            # 1-lot buys (cheap talk)
            one_lot_buy_count[b] = np.sum((dc[mask] == 1) & (pf[mask] == 1))
            # Large sells (real action)
            ls_mask = mask & (pf == 0) & (dc >= lt)
            large_sell_vol[b] = dc[ls_mask].sum()
        
        # Ratio: large sell per 1-lot noise
        raw_bins = large_sell_vol / (one_lot_buy_count + EPS)
        daily_raw = np.mean(raw_bins)
        results.append({'Date': date, 'raw': daily_raw})
    
    daily = pd.DataFrame(results)
    if daily.empty:
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_cheap_talk_distraction_trap'])
    
    daily = daily.sort_values('Date').reset_index(drop=True)
    out = _zscore_rolling(daily['raw'], 20)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    
    return pd.DataFrame({
        'StockId': stock_id,
        'Date': daily['Date'],
        'f_gt_cheap_talk_distraction_trap': out.values
    })
