import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import ewm_then_zscore


@register_feature
class FeatureGtHeterogeneousStigmaBlending(BaseFeature):
    name = "f_gt_heterogeneous_stigma_blending"
    description = "污名混合分散 - 賣方集中度低且賣出量大時"
    required_columns = ["StockId", "Date", "raw_hhi_sell", "raw_top5_sell", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # HHI sell already computed: sum((SellQtm/total_sell)^2)
        # blending = 1.0 / (hhi_sell + 1e-5)
        blending = 1.0 / (df["raw_hhi_sell"] + 1e-5)
        
        sell_mean = df.groupby("StockId")["raw_top5_sell"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        inst_dump = (df["raw_top5_sell"] > sell_mean * 1.5).astype(int)
        
        raw = inst_dump * blending
        
        out = ewm_then_zscore(raw, ewm_span=5, z_window=20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
