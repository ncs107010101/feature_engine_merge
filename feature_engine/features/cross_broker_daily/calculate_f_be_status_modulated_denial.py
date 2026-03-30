import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore


@register_feature
class FeatureBeStatusModulatedDenial(BaseFeature):
    name = "f_be_status_modulated_denial"
    description = "地位調節否認 - 跳空下跌時主力隱瞞賣出"
    required_columns = ["StockId", "Date", "raw_top5_net_buy_nlargest", "收盤價", "開盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # prev_close = previous day close
        prev_close = df.groupby("StockId")["收盤價"].shift(1)
        
        # gap_down = int(open < prev_close * 0.98)
        gap_down = (df["開盤價"] < prev_close * 0.98).astype(int)
        
        # Get top5 brokers by total_vol (lookback 60)
        # top5_ns = -(top5 NetBuy sum)
        top5_ns = df["raw_top5_net_buy_nlargest"].apply(lambda x: max(0, -x))
        
        # rest_nb = rest NetBuy sum (approximated)
        rest_nb = -df["raw_top5_net_buy_nlargest"]
        
        # total_vol = sum(BuyQtm + SellQtm) - approximated as 2 * abs(top5_net_buy)
        total_vol = 2 * np.abs(df["raw_top5_net_buy_nlargest"])
        
        rest_nb_norm = rest_nb / (total_vol + 1e-5)
        
        val = gap_down * (top5_ns > 0).astype(int) * (rest_nb > 0).astype(int) * rest_nb_norm
        
        out = ewm_then_zscore(val, ewm_span=5, z_window=20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
