import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureCrackLiquidityCollapse(BaseFeature):
    name = "f_crack_liquidity_collapse"
    description = (
        "基態重力塌縮指標：比較日內最佳買/賣價的最大連續移動方向差異。"
        "50-tick bins的BuyPr/SellPr相對變動，正值=賣方撤退>買方撤退→看漲。"
    )
    required_columns = ["StockId", "Date", "raw_crack_liquidity_collapse"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_crack_liquidity_collapse"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
