import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import phi_coefficient, EPS

WINDOW = 20
MIN_OBS = 10

@register_feature
class FItCausalPhiCoefficient(BaseFeature):
    name = "f_it_causal_phi_coefficient"
    description = "Causal phi coefficient: Phi coefficient between broker direction and tick direction. Range [-1, +1]. Measures correlation between large broker net buying and active buy ratio."
    required_columns = [
        "StockId", "Date", "raw_big_net_ratio", "raw_p_active_up"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        all_features["B_t"] = (all_features["raw_big_net_ratio"] > 0).astype(int)
        all_features["Y_t"] = (all_features["raw_p_active_up"] > 0.5).astype(int)

        n = len(all_features)
        f06 = np.full(n, np.nan)

        B = all_features["B_t"].values
        Y = all_features["Y_t"].values

        for i in range(WINDOW, n):
            win_b = B[i - WINDOW:i]
            win_y = Y[i - WINDOW:i]
            if (win_b.sum() < 2) or (win_b.sum() > WINDOW - 2):
                continue
            if (win_y.sum() < 2) or (win_y.sum() > WINDOW - 2):
                continue

            f06[i] = phi_coefficient(win_b, win_y)

        out = pd.Series(f06, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
