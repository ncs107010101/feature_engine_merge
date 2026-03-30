import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeaturePriceZoneMigration(BaseFeature):
    name = "f_price_migration"
    description = "買賣區間遷移距離 (Price Zone Migration Distance)"
    required_columns = ["StockId", "Date", "收盤價", "最高價", "最低價", "_atr_20"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        if "成交金額(元)" in df.columns and "成交量(千股)" in df.columns:
            price = df["成交金額(元)"] / (df["成交量(千股)"] * 1000 + 1e-8)
            price = price.replace([np.inf, -np.inf], np.nan).fillna(df["收盤價"])
        else:
            price = (df["最高價"] + df["最低價"] + df["收盤價"] * 2) / 4
            
        # 5日遷移距離
        migration = price.groupby(df["StockId"]).diff().abs().groupby(df["StockId"]).rolling(window=5, min_periods=1).sum().reset_index(0, drop=True)
        
        # ATR 20
        atr = df["_atr_20"]
        
        migration_norm = migration / (atr + 1e-8)
        
        result = migration_norm.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=1))
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
