import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

WINDOW = 20
EPS = 1e-9


def compute_f19_causal_transfer_rate_deviation(di_series, ewm_today):
    valid = di_series[~np.isnan(di_series)]
    if len(valid) < 5:
        return np.nan
    t = np.arange(len(valid), dtype=float)
    if np.std(t) < EPS:
        return np.nan
    slope = np.polyfit(t, valid, 1)[0]
    return float(slope * ewm_today)


@register_feature
class FItCausalTransferRateDeviation(BaseFeature):
    name = "f_it_causal_transfer_rate_deviation"
    description = "Causal Transfer Rate Deviation: DI series slope × ewm_net. Measures the trend of directional information flow strength between broker and tick."
    required_columns = [
        "StockId", "Date", "raw_di_series", "raw_ewm_net"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()
        all_features = all_features.sort_values("Date")

        di_arr = all_features["raw_di_series"].values
        ewm_arr = all_features["raw_ewm_net"].values

        n = len(all_features)
        f19 = np.full(n, np.nan)

        all_features_sorted = all_features.reset_index()
        stock_groups = all_features_sorted.groupby("StockId")

        for stock_id, group in stock_groups:
            indices = group.index.tolist()
            di_stock = di_arr[indices]
            ewm_stock = ewm_arr[indices]
            for i in range(WINDOW, len(indices)):
                if np.isnan(di_stock[i]):
                    continue
                di_win = di_stock[i - WINDOW:i]
                ewm_today = ewm_stock[i]
                f19[indices[i]] = compute_f19_causal_transfer_rate_deviation(di_win, ewm_today)

        out = pd.Series(f19, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
