import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature

@register_feature
class FeatureNetMarkovFlowAsymmetry(BaseFeature):
    name = "f_net_markov_flow_asymmetry"
    description = (
        "馬可夫鏈流非對稱：計算量加權的上漲 vs 下跌位移之差，"
        "標準化為 (ΣVol_up×|ΔP_up| - ΣVol_dn×|ΔP_dn|) / total。"
        "正值=向上流動佔優。"
    )
    required_columns = ["StockId", "Date", "raw_net_markov_flow_asymmetry"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        all_f = df.set_index(["StockId", "Date"]).sort_index()
        out = all_f["raw_net_markov_flow_asymmetry"]
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({self.name: out}).reset_index()
