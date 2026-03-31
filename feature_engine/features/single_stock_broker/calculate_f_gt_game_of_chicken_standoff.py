import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FGtGameOfChickenStandoff(BaseFeature):
    name = "f_gt_game_of_chicken_standoff"
    description = "Game Theory v17: Buy HHI and Sell HHI both extremely high (90th percentile, 60-day) with tiny difference → standoff. Positive → Extreme standoff, high volatility expected → Direction uncertain"
    required_columns = ["StockId", "Date", "raw_hhi_buy", "raw_hhi_sell"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        EPS = 1e-8
        
        hhi_buy = df["raw_hhi_buy"].values.astype(float)
        hhi_sell = df["raw_hhi_sell"].values.astype(float)
        
        buy_q90 = df.groupby("StockId")["raw_hhi_buy"].transform(
            lambda x: x.rolling(60, min_periods=20).quantile(0.9).shift(1)
        ).values.astype(float)
        sell_q90 = df.groupby("StockId")["raw_hhi_sell"].transform(
            lambda x: x.rolling(60, min_periods=20).quantile(0.9).shift(1)
        ).values.astype(float)
        
        is_high_buy = (hhi_buy > buy_q90).astype(int)
        is_high_sell = (hhi_sell > sell_q90).astype(int)
        
        standoff = 1.0 / (np.abs(hhi_buy - hhi_sell) + EPS)
        raw = is_high_buy * is_high_sell * standoff
        
        out = pd.Series(raw, index=df.index).groupby(df["StockId"]).transform(
            lambda x: x.ewm(span=3, min_periods=1).mean()
        )
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
