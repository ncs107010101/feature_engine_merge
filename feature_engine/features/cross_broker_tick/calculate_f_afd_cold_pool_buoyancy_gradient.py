import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureAFDColdPoolBuoyancyGradient(BaseFeature):
    name = "f_afd_cold_pool_buoyancy_gradient"
    description = "Large investor dynamic lift overcoming retail negative buoyancy"
    required_columns = ["StockId", "Date", "raw_top5_net_buy", "raw_retail_net_buy", "raw_delta_p"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        # Step 1: u_dyn = abs(raw_top5_net_buy)
        u_dyn = df["raw_top5_net_buy"].abs()

        # Step 2: b_cold = abs(minimum(0, raw_retail_net_buy))
        b_cold = df["raw_retail_net_buy"].clip(upper=0).abs()

        # Step 3: dir_sign = sign(raw_delta_p)
        dir_sign = np.sign(df["raw_delta_p"])

        # Step 4: raw = maximum(0, u_dyn - b_cold) * dir_sign
        raw = (u_dyn - b_cold).clip(lower=0) * dir_sign

        # Step 5: zscore_rolling with window=20, eps=1e-10
        out_series = raw.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, window=20, eps=1e-10)
        )

        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })