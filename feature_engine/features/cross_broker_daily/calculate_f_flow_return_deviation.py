import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureFlowReturnDeviation(BaseFeature):
    name = "f_flow_deviation"
    description = "資金流向與報酬偏離度 (Flow-Return Deviation)"
    required_columns = ["StockId", "Date", "_total_net", "_daily_return"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(["StockId", "Date"])
        
        net_flow = df["_total_net"].fillna(0)
        ret = df["_daily_return"]
        
        def _deviation(group):
            ret_g = group["ret"]
            flow_g = group["flow"]
            cov = flow_g.rolling(window=60, min_periods=10).cov(ret_g)
            var = ret_g.rolling(window=60, min_periods=10).var()
            beta = cov / (var + 1e-8)
            
            flow_mean = flow_g.rolling(window=60, min_periods=10).mean()
            ret_mean = ret_g.rolling(window=60, min_periods=10).mean()
            alpha = flow_mean - beta * ret_mean
            
            expected_flow = alpha + beta * ret_g
            residual = flow_g - expected_flow
            
            res_std = residual.rolling(window=60, min_periods=10).std()
            dev = residual / (res_std + 1e-8)
            return dev.clip(-5, 5)
            
        tmp = pd.DataFrame({"ret": ret, "flow": net_flow, "StockId": df["StockId"]})
        
        devs = []
        for _, group in tmp.groupby("StockId"):
            devs.append(_deviation(group))
        
        result = pd.concat(devs).sort_index().fillna(0)
        
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({"StockId": df["StockId"], "Date": df["Date"], self.name: result})
