import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import (
    rolling_binary_sequences, compute_causal_kelly_growth_vectorized,
    WINDOW, MIN_OBS
)

@register_feature
class FItCausalKellyGrowth(BaseFeature):
    name = "f_it_causal_kelly_growth"
    description = "因果凱利增量: f* - f_base, measuring causal Kelly growth (20-day window)"
    required_columns = ["StockId", "Date", "raw_big_net_ratio", "raw_p_active_up"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = rolling_binary_sequences(data.copy())
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        results = []
        for stock_id, grp in df.groupby("StockId"):
            grp = grp.sort_values("Date").reset_index(drop=True)
            B = grp["B_seq"].values
            Y = grp["Y_seq"].values
            n = len(grp)
            
            kelly = np.full(n, np.nan)
            for i in range(WINDOW, n):
                win_b = B[i - WINDOW:i]
                win_y = Y[i - WINDOW:i]
                b_today = B[i]
                
                if win_b.sum() < 2 or win_b.sum() > WINDOW - 2:
                    continue
                kelly[i] = compute_causal_kelly_growth_vectorized(win_b, win_y, b_today)
            
            grp_result = pd.DataFrame({
                "StockId": stock_id,
                "Date": grp["Date"].values,
                self.name: kelly
            })
            results.append(grp_result)
        
        return pd.concat(results, ignore_index=True)
