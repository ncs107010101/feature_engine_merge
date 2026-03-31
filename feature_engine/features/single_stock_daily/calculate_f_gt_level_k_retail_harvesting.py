import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureGtLevelKRetailHarvesting(BaseFeature):
    name = "f_gt_level_k_retail_harvesting"
    description = "Game Theory Module 3: Level-k retail harvesting. When day-trade heat correlates with returns but institutions go opposite direction."
    required_columns = ["StockId", "Date", "_dt_pct", "_ret_5d", "_inst_net"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        corr_list = []
        for stock_id, group in df.groupby("StockId", sort=False):
            g = group.reset_index(drop=True)
            corr = g["_dt_pct"].rolling(10, min_periods=5).corr(g["_ret_5d"])
            corr_list.append(corr.values)
        
        corr_all = np.concatenate(corr_list)
        
        raw = pd.Series(corr_all, index=df.index) * (-np.sign(df["_inst_net"].values)) * np.abs(df["_inst_net"].values)
        
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
