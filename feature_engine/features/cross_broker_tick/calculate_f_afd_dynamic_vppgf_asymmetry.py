import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureAFDDynamicVppgfAsymmetry(BaseFeature):
    name = "f_afd_dynamic_vppgf_asymmetry"
    description = "Large momentum squared vs retail momentum squared"
    required_columns = ["StockId", "Date", "raw_top10_net_buy", "raw_retail_net_buy", "raw_delta_p"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        zeta_large = df["raw_top10_net_buy"]
        zeta_small = df["raw_retail_net_buy"]
        raw = np.maximum(0, zeta_large**2 - zeta_small**2) * df["raw_delta_p"]
        raw = np.log1p(np.abs(raw)) * np.sign(raw)

        out_series = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=42, eps=1e-10))
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: out})