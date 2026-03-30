import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyBrightDarkModeCovariance(BaseFeature):
    name = "f_phy_bright_dark_mode_covariance"
    description = "亮暗模協方差：大券商淨買←→散戶淨買的20日滾動相關×大券商比率的負值，衡量主力與散戶的背離強度。"
    required_columns = ["StockId", "Date", "raw_phy_bright_dark_mode_covariance"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_bright_dark_mode_covariance"]]
            .copy()
            .rename(columns={"raw_phy_bright_dark_mode_covariance": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
