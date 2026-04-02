import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureAFDMoistCiskFeedback(BaseFeature):
    name = "f_afd_moist_cisk_feedback"
    description = "CISK feedback - 5-day rolling covariance of ret and large_net times net diff"
    required_columns = ["StockId", "Date", "raw_tick_ret", "raw_top10_net_buy", "raw_retail_net_buy"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        net_diff = (df["raw_top10_net_buy"] - df["raw_retail_net_buy"]).clip(lower=0)

        def rolling_cov(x, y, window=5):
            return x.rolling(window, min_periods=1).cov(y)

        cov = df["raw_tick_ret"].groupby(df["StockId"]).transform(
            lambda x: rolling_cov(x, df.loc[x.index, "raw_top10_net_buy"], window=5)
        )

        raw = cov * net_diff
        raw = np.log1p(np.abs(raw)) * np.sign(raw)
        out = raw.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=42, eps=1e-10))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: out})
