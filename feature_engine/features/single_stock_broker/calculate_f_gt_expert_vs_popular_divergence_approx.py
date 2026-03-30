import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FGtExpertVsPopularDivergenceApprox(BaseFeature):
    """
    Game Theory v17: Expert vs Popular Divergence (Approximation)

    Original logic: Popular brokers (high 5d volume) bullish but Expert brokers
    (60d net accumulators) bearish. Positive → Popular bullish but experts exit → Extreme LOW return.

    Approximation using daily-level aggregates (raw_top5_net_buy, raw_mid10_net_sell) - not identical
    to original broker-day rolling algorithm.

    Signal: Top-5 volume brokers are buying (bullish) while mid-tier brokers are selling (bearish).
    This represents divergence between high-activity and mid-tier broker behavior.
    """
    name = "f_gt_expert_vs_popular_divergence_approx"
    description = "Game Theory v17 (Approx): Popular brokers (top 5 by buy volume) bullish but mid-tier brokers (rank 6-15) bearish. Approximation using daily-level aggregates (raw_top5_net_buy, raw_mid10_net_sell) - not identical to original broker-day rolling algorithm. Positive → Divergence between high-activity and mid-tier brokers → Extreme LOW return"
    required_columns = ["StockId", "Date", "raw_top5_net_buy", "raw_mid10_net_sell"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)

        # Approximation using daily-level aggregates:
        # - raw_top5_net_buy: Top 5 brokers by buy VOLUME (positive net buy)
        # - raw_mid10_net_sell: Mid 10 brokers by buy VOLUME (negative net buy)
        # Divergence: popular (top 5) is buying AND mid-tier is selling

        top5_net = df["raw_top5_net_buy"]
        mid10_net = df["raw_mid10_net_sell"]

        # Signal: top-5 is positive AND mid-10 is positive (meaning mid-10 is net selling)
        is_diverge = (top5_net > 0).astype(float) * (mid10_net > 0).astype(float)
        magnitude = np.abs(top5_net - mid10_net)

        raw_signal = is_diverge * magnitude

        # Apply EWM then zscore (same as other features)
        ewm = raw_signal.groupby(df["StockId"]).transform(lambda x: x.ewm(span=5, min_periods=1).mean())
        out = ewm.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, 20))
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)

        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
