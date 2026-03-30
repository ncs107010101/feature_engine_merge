import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureQitLiquidityCatalysisRatio(BaseFeature):
    name = "f_qit_liquidity_catalysis_ratio"
    description = "流動性催化率。淨買賣極小但成交量大的造市券商成佔比。"
    
    required_columns = ["StockId", "Date", "raw_qit_liquidity_catalysis_ratio"]
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data[["StockId", "Date", "raw_qit_liquidity_catalysis_ratio"]].copy()
        df.rename(columns={"raw_qit_liquidity_catalysis_ratio": self.name}, inplace=True)
        return df
