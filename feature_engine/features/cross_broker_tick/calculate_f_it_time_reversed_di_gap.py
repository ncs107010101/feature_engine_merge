import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import directed_info_1markov, EPS

WINDOW = 20
MIN_OBS = 10

@register_feature
class FItTimeReversedDiGap(BaseFeature):
    name = "f_it_time_reversed_di_gap"
    description = "Time-reversed Directed Information gap: (DI_fwd - DI_rev) / (DI_fwd + DI_rev) * ewm_net. Measures irreversibility in information flow between broker and tick, weighted by momentum."
    required_columns = [
        "StockId", "Date", "raw_big_net_ratio", "raw_p_active_up", "raw_ewm_net"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        all_features["B_t"] = (all_features["raw_big_net_ratio"] > 0).astype(int)
        all_features["Y_t"] = (all_features["raw_p_active_up"] > 0.5).astype(int)

        n = len(all_features)
        f08 = np.full(n, np.nan)

        B = all_features["B_t"].values
        Y = all_features["Y_t"].values
        ewm = all_features["raw_ewm_net"].values

        for i in range(WINDOW, n):
            win_b = B[i - WINDOW:i]
            win_y = Y[i - WINDOW:i]
            if (win_b.sum() < 2) or (win_b.sum() > WINDOW - 2):
                continue
            if (win_y.sum() < 2) or (win_y.sum() > WINDOW - 2):
                continue

            di_fwd = directed_info_1markov(win_b, win_y)
            di_rev = directed_info_1markov(win_b[::-1], win_y[::-1])
            denom = di_fwd + di_rev + EPS
            irreversibility = (di_fwd - di_rev) / denom
            f08[i] = irreversibility * ewm[i]

        out = pd.Series(f08, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
