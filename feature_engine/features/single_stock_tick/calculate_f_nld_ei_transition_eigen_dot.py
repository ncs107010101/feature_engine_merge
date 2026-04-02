import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import (build_transition_matrix, stationary_distribution,
                      prflag_to_ei_states, zscore_rolling, safe_clip_fillna)


@register_feature
class FeatureNldEiTransitionEigenDot(BaseFeature):
    """
    特徵名: f_nld_ei_transition_eigen_dot
    靈感來源: E/I 狀態轉移矩陣特徵向量 × 價格位移內積
    計算邏輯: PrFlag 構建 E/I 2×2 轉移矩陣 → 主特徵向量 · [ΔP_E, ΔP_I]
    """
    name = "f_nld_ei_transition_eigen_dot"
    description = "E/I轉移特徵內積：E/I狀態轉移矩陣的平穩分佈與價格位移的內積，衡量狀態機與價格動量的對齊"
    required_columns = ["StockId", "Date"]
    data_combination = "single_stock_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tick_raw = kwargs.get('_tick_raw')
        if tick_raw is None:
            raise ValueError("This feature requires raw tick data in kwargs['_tick_raw']")
        df = tick_raw.copy()
        df = df.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
        
        results = []
        for (stock_id, date), day_df in df.groupby(["StockId", "Date"]):
            if len(day_df) < 30:
                results.append({"StockId": stock_id, "Date": date, "raw": 0.0})
                continue
            
            prices = day_df["DealPrice"].values
            prflag = day_df["PrFlag"].values
            
            ei_states = prflag_to_ei_states(prflag)
            tm = build_transition_matrix(ei_states, n_states=2, laplace_smooth=1.0)
            pi = stationary_distribution(tm)
            
            # 計算 E 和 I 狀態下的價格位移
            dp = np.diff(prices)
            ei_for_dp = ei_states[1:]  # offset by 1
            
            dp_e = dp[ei_for_dp == 0]
            dp_i = dp[ei_for_dp == 1]
            
            mean_dp_e = np.mean(dp_e) if len(dp_e) > 0 else 0.0
            mean_dp_i = np.mean(dp_i) if len(dp_i) > 0 else 0.0
            
            if not (np.isfinite(mean_dp_e) and np.isfinite(mean_dp_i)):
                results.append({"StockId": stock_id, "Date": date, "raw": 0.0})
                continue
            
            price_vec = np.array([mean_dp_e, mean_dp_i])
            val = np.dot(pi, price_vec)
            
            results.append({"StockId": stock_id, "Date": date, "raw": val})
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        result_df[self.name] = result_df.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20, eps=1e-10)
        )
        result_df[self.name] = safe_clip_fillna(result_df[self.name])
        
        return result_df[["StockId", "Date", self.name]]
