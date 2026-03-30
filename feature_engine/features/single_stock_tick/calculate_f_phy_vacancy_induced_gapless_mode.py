import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyVacancyInducedGaplessMode(BaseFeature):
    name = "f_phy_vacancy_induced_gapless_mode"
    description = "空位誘導的無隙模式：量加權的跳躍幅度平方和，捕捉異常放量時的劇烈價格波動能量。"
    required_columns = ["StockId", "Date", "raw_phy_vacancy_induced_gapless_mode"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_vacancy_induced_gapless_mode"]]
            .copy()
            .rename(columns={"raw_phy_vacancy_induced_gapless_mode": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
