import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeaturePhyGeodesicJacobiConvergence(BaseFeature):
    name = "f_phy_geodesic_jacobi_convergence"
    description = "測地線 Jacobi 收斂：主力買入成本優勢×淨買方向，衡量主力相對散戶的成本優勢強度。若無成交金額欄位則用報酬×淨買代替。"
    required_columns = ["StockId", "Date", "raw_phy_geodesic_jacobi_convergence"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = (
            data[["StockId", "Date", "raw_phy_geodesic_jacobi_convergence"]]
            .copy()
            .rename(columns={"raw_phy_geodesic_jacobi_convergence": self.name})
        )
        out[self.name] = out[self.name].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
