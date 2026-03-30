import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import kl_divergence

EPS = 1e-3

@register_feature
class FItPublicPrivateKlDirection(BaseFeature):
    name = "f_it_public_private_kl_direction"
    description = "Wiretap channel: Public-private message divergence. KL(Public 2-bin || Private 2-bin) * direction"
    required_columns = [
        "StockId", "Date", "raw_p_active_up", "raw_p_public_buy"
    ]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_features = df.set_index(["StockId", "Date"]).sort_index()

        results = []
        for idx in all_features.index:
            p_public_buy = all_features.loc[idx, "raw_p_public_buy"]
            p_private_buy = all_features.loc[idx, "raw_p_active_up"]

            if pd.isna(p_public_buy) or pd.isna(p_private_buy):
                results.append(np.nan)
                continue

            p_pub = np.array([1 - p_public_buy, p_public_buy]) + EPS
            p_pub /= p_pub.sum()
            p_pri = np.array([1 - p_private_buy, p_private_buy]) + EPS
            p_pri /= p_pri.sum()

            kl = kl_divergence(p_pub, p_pri)
            direction = (p_private_buy - 0.5) * 2.0

            val = kl * direction
            results.append(val)

        out = pd.Series(results, index=all_features.index)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        final_result = pd.DataFrame({
            self.name: out
        }).reset_index()

        return final_result
