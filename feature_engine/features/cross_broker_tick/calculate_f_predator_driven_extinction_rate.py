import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FPredatorDrivenExtinctionRate(BaseFeature):
    name = "f_predator_driven_extinction_rate"
    description = "Top5_Sell/PassiveBuy → pct_change(5) → zscore(42). Top 5券商賣出量相對於日內被動承接買量的比率。"
    required_columns = ["StockId", "Date", "raw_top5_sell_vol", "raw_passive_buy_vol"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        EPS = 1e-9
        
        # Ensure proper time-series ordering
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        
        # Compute ratio: Top5 sell / Passive buy
        ratio = all_features["raw_top5_sell_vol"] / (all_features["raw_passive_buy_vol"] + EPS)
        
        # pct_change(5) per stock - MUST use groupby to avoid cross-stock contamination
        pct_chg = ratio.groupby(level="StockId").pct_change(5)
        
        # Handle inf from pct_change
        pct_chg = pct_chg.replace([np.inf, -np.inf], np.nan)
        
        # Rolling z-score with window=42
        out = pct_chg.groupby(level="StockId").transform(
            lambda x: zscore_rolling(x, window=42, min_periods=21)
        )
        
        # Clean up inf/nan
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({
            self.name: out
        }).reset_index()
