import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureHeatReturnLag(BaseFeature):
    name = "f_heat_lag"
    description = "熱度-報酬滯後係數 (Heat-Return Lag Coefficient)"
    required_columns = ["StockId", "Date", "收盤價", "最高價", "最低價", "成交量(千股)", "流通在外股數(千股)", "_atr_5"]
    data_combination = "single_stock_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        vol = df["成交量(千股)"]
        shares = df["流通在外股數(千股)"].fillna(1)
        turnover = (vol / shares.replace(0, 1)) * 100
        
        day_trade = df.get("當沖買賣比", pd.Series(0, index=df.index)).fillna(0)
        
        atr_5 = df["_atr_5"]
        volatility = (atr_5 / (df["收盤價"] + 1e-8)) * 100
        
        # Normalize each component individually before combining (fix mixed-dimension issue)
        z_turnover = turnover.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=5))
        z_daytrade = day_trade.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=5))
        z_volatility = volatility.groupby(df["StockId"]).transform(lambda x: zscore_rolling(x, window=20, min_periods=5))
        
        heat = z_turnover + z_daytrade + z_volatility
        ret = df.groupby("StockId")["收盤價"].pct_change()
        
        heat_lag = heat.groupby(df["StockId"]).shift(1)
        
        # rolling corr between heat_lag and ret
        tmp = pd.DataFrame({"heat_lag": heat_lag, "ret": ret, "StockId": df["StockId"]})
        corrs = []
        for _, group in tmp.groupby("StockId"):
            c = group["heat_lag"].rolling(window=20, min_periods=5).corr(group["ret"])
            corrs.append(c)
        lag_corr = pd.concat(corrs).sort_index().fillna(0)
        
        result = lag_corr.clip(-1, 1)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
