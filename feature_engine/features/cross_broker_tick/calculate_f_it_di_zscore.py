import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import directed_info_1markov, rolling_zscore, EPS

WINDOW = 20
MIN_OBS = 10


@register_feature
class FItDiZscore(BaseFeature):
    name = "f_it_di_zscore"
    description = "Directed Information Z-score: rolling z-score of 1st order directed information I(B->Y) multiplied by ewm of net_ratio_big. Measures directional information flow between broker and tick with time-weighted normalization."
    required_columns = [
        "StockId", "Date", "raw_big_net_ratio", "raw_p_active_up", "raw_ewm_net"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        all_features["B_seq"] = (all_features["raw_big_net_ratio"] > 0).astype(int)
        all_features["Y_seq"] = (all_features["raw_p_active_up"] > 0.5).astype(int)

        n = len(all_features)
        di_series = np.full(n, np.nan)
        f17 = np.full(n, np.nan)

        B = all_features["B_seq"].values
        Y = all_features["Y_seq"].values
        ewm_arr = all_features["raw_ewm_net"].values

        for i in range(WINDOW, n):
            win_b = B[i - WINDOW:i]
            win_y = Y[i - WINDOW:i]

            if (win_b.sum() < 2) or (win_b.sum() > WINDOW - 2):
                continue
            if (win_y.sum() < 2) or (win_y.sum() > WINDOW - 2):
                continue

            di_val = directed_info_1markov(win_b, win_y)
            di_series[i] = di_val

        di_pd = pd.Series(di_series, index=all_features.index)
        di_z = rolling_zscore(di_pd, WINDOW)
        ewm_series = pd.Series(ewm_arr, index=all_features.index)
        f17 = (di_z * ewm_series).values

        out = pd.Series(f17, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
