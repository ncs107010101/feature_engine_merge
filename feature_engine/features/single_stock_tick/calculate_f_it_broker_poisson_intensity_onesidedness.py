import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureITBrokerPoissonIntensityOnesidedness(BaseFeature):
    name = "f_it_broker_poisson_intensity_onesidedness"
    description = (
        "15分鐘時窗計算大額成交的買/賣方泊松強度。"
        "特徵值 = (λ_buy - λ_sell) / (λ_buy + λ_sell + ε)，範圍[-1, +1]。"
    )
    required_columns = ["StockId", "Date", "raw_it_broker_poisson_intensity_onesidedness"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_it_broker_poisson_intensity_onesidedness"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
