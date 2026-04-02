import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FBeAsymmetricConfidenceSpillover(BaseFeature):
    name = "f_be_asymmetric_confidence_spillover"
    description = "Behavioral Economics: Asymmetric confidence spillover. When price drops but day-trade ratio does not shrink proportionally, indicating retail overconfidence."
    required_columns = ["StockId", "Date", "_dt_pct", "_dt_mean_20", "_ret_3d"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        g = df.groupby(df["StockId"])
        
        dt_trend = df["_dt_pct"] / (df["_dt_mean_20"] + 1e-5)
        
        bad_news = (df["_ret_3d"] < -0.03).astype(int)
        
        raw = dt_trend * bad_news
        
        out = zscore_rolling(raw, 20)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
