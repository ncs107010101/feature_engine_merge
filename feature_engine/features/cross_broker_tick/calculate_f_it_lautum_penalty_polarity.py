import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import (
    rolling_binary_sequences, compute_lautum_penalty_vectorized,
    WINDOW, MIN_OBS
)

@register_feature
class FItLautumPenaltyPolarity(BaseFeature):
    name = "f_it_lautum_penalty_polarity"
    description = "Lautum penalty polarity: -D_KL(P(B|Y_lag)||P(B)) * net_ratio (20-day window)"
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
            net_ratio = grp["raw_big_net_ratio"].values
            n = len(grp)
            
            lautum = np.full(n, np.nan)
            for i in range(WINDOW, n):
                win_b = B[i - WINDOW:i]
                win_y = Y[i - WINDOW:i]
                net_today = net_ratio[i]
                
                if win_b.sum() < 2:
                    continue
                lautum[i] = compute_lautum_penalty_vectorized(win_b, win_y, net_today)
            
            grp_result = pd.DataFrame({
                "StockId": stock_id,
                "Date": grp["Date"].values,
                self.name: lautum
            })
            results.append(grp_result)
        
        return pd.concat(results, ignore_index=True)
