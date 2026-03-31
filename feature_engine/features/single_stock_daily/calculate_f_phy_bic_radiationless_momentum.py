import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyBicRadiationlessMomentum(BaseFeature):
    name = "f_phy_bic_radiationless_momentum"
    description = "BIC 無輻射動量：成交量比×高低幅倒數×日內方向，衡量在低波動日放量帶來的定向動量。"
    required_columns = ["StockId", "Date", "raw_phy_bic_radiationless_momentum"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_bic_radiationless_momentum"]]
            .copy()
            .rename(columns={"raw_phy_bic_radiationless_momentum": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
