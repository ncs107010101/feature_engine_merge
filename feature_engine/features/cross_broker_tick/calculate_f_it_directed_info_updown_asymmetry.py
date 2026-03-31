import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import directed_info_split, rolling_zscore, EPS

WINDOW = 20
MIN_OBS = 10

@register_feature
class FItDirectedInfoUpdownAsymmetry(BaseFeature):
    name = "f_it_directed_info_updown_asymmetry"
    description = "Directed Information up/down asymmetry: DI_up - DI_down. Measures directional information asymmetry between broker flow and tick activity. Standardized with rolling z-score."
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
        f05 = np.full(n, np.nan)

        B = all_features["B_t"].values
        Y = all_features["Y_t"].values

        for i in range(WINDOW, n):
            win_b = B[i - WINDOW:i]
            win_y = Y[i - WINDOW:i]
            if (win_b.sum() < 2) or (win_b.sum() > WINDOW - 2):
                continue
            if (win_y.sum() < 2) or (win_y.sum() > WINDOW - 2):
                continue

            di_up, di_dn = directed_info_split(win_b, win_y)
            f05[i] = di_up - di_dn

        raw_series = pd.Series(f05, index=all_features.index)
        
        # Apply rolling z-score (this fills values after WINDOW days)
        zscore_applied = raw_series.groupby(level="StockId").transform(
            lambda x: rolling_zscore(x, WINDOW)
        )
        
        # Set first WINDOW rows to NaN (not enough history)
        out = zscore_applied.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
