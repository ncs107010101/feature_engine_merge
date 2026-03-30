import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature


@register_feature
class FeatureIncumbentSignalingExemption(BaseFeature):
    name = "f_gt_incumbent_signaling_exemption"
    description = "Incumbent Signaling Exemption (現任者豁免): Top1 incumbent from 10-day history has passive buying. Dominant broker's passive accumulation signals hidden conviction."
    required_columns = [
        "StockId", "Date",
        "top1_10d_netbuy", "top1_10d_incumbent_buy",
        "收盤價", "開盤價"
    ]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        # passive_proxy: incumbent buy / rolling mean incumbent buy
        inc_roll_mean = df.groupby("StockId")["top1_10d_incumbent_buy"].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
        )
        passive_proxy = df["top1_10d_incumbent_buy"] / (inc_roll_mean + 1e-5)

        # abs_ret: absolute daily return
        abs_ret = np.abs(df["收盤價"] - df["開盤價"]) / (df["開盤價"] + 1e-8)

        # Raw signal: passive behavior relative to price movement
        raw = 1.0 / (abs_ret / (passive_proxy + 1e-5) + 1e-5)
        raw = np.clip(raw, -1e6, 1e6)

        # EWM then z-score
        def ewm_then_zscore(x, span=20):
            ewm = x.shift(1).ewm(span=span, adjust=False).mean()
            mean = ewm.rolling(window=span, min_periods=5).mean()
            std = ewm.rolling(window=span, min_periods=5).std()
            return (ewm - mean) / (std + 1e-8)

        out = ewm_then_zscore(pd.Series(raw.values), span=5)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })