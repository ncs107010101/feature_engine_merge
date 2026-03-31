"""
Feature: f_gt_invidious_distinction_cost
Module: Game Theory v17 - Module 4
Theory: Invidious Distinction - Large buyers pay premium above VWAP, demonstrating information superiority
Direction: Positive → Premium buying signals strong bullish intent → Extreme HIGH return
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
class FeatureGtInvidiousDistinctionCost(BaseFeature):
    name = "f_gt_invidious_distinction_cost"
    description = "Game Theory Module 4: Invidious Distinction - Large buyers pay premium above VWAP, demonstrating information superiority. Premium buying signals strong bullish intent and extreme HIGH return."
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            raise ValueError("_tick_raw is required for f_gt_invidious_distinction_cost")
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
        return pd.DataFrame(columns=['StockId', 'Date', 'f_gt_invidious_distinction_cost'])
    
    FEATURE_NAME = 'f_gt_invidious_distinction_cost'
    BIN_SIZE = 500
    
    # Threshold
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
        if n < BIN_SIZE:
            results.append({'Date': date, 'raw': 0.0})
            continue
        
        dc = day_ticks['DealCount'].values.astype(float)
        dp = day_ticks['DealPrice'].values.astype(float)
        pf = day_ticks['PrFlag'].values
        
        bin_ids = np.arange(n) // BIN_SIZE
        n_bins = bin_ids[-1] + 1
        
        raw_bins = []
        for b in range(n_bins):
            mask = bin_ids == b
            dc_b = dc[mask]
            dp_b = dp[mask]
            pf_b = pf[mask]
            
            # Bin VWAP
            total_val = np.sum(dp_b * dc_b)
            total_vol = np.sum(dc_b)
            if total_vol == 0:
                continue
            bin_vwap = total_val / total_vol
            
            # Large active buy VWAP
            lb_mask = (pf_b == 1) & (dc_b >= lt)
            if np.sum(lb_mask) == 0:
                raw_bins.append(0.0)
                continue
            lb_val = np.sum(dp_b[lb_mask] * dc_b[lb_mask])
            lb_vol = np.sum(dc_b[lb_mask])
            lb_vwap = lb_val / lb_vol
            
            # Premium rate × volume
            premium = (lb_vwap / bin_vwap - 1.0) * lb_vol
            raw_bins.append(premium)
        
        daily_raw = np.mean(raw_bins) if raw_bins else 0.0
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