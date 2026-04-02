import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature
from ...utils import safe_second_derivative, zscore_rolling, safe_clip_fillna


@register_feature
class FeatureNldSynapticDecayCrossDot(BaseFeature):
    """
    特徵名: f_nld_synaptic_decay_cross_dot
    靈感來源: 突觸衰減時間常數差異 — 快/慢動態的交叉內積
    計算邏輯: E快速平滑 vs I慢速平滑 → 各自二階導 → 內積 × AsymSign
    """
    name = "f_nld_synaptic_decay_cross_dot"
    description = "突觸衰減交叉內積：快/慢EWM平滑的二階導內積，衡量多空動態的時間尺度耦合"
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
            if len(day_df) < 50:
                results.append({"StockId": stock_id, "Date": date, "raw": 0.0})
                continue
            
            volumes = day_df["DealCount"].values.astype(float)
            prflag = day_df["PrFlag"].values
            prices = day_df["DealPrice"].values
            
            # E (外盤/主動買) 成交量序列
            e_vol = np.where(prflag == 1, volumes, 0).astype(float)
            i_vol = np.where(prflag == 0, volumes, 0).astype(float)
            
            # 快速平滑 E (span=5), 慢速平滑 I (span=20)
            e_fast = pd.Series(e_vol).ewm(span=5, min_periods=1).mean()
            i_slow = pd.Series(i_vol).ewm(span=20, min_periods=1).mean()
            
            # 各自二階導數
            d2e = safe_second_derivative(e_fast, ewm_span=5)
            d2i = safe_second_derivative(i_slow, ewm_span=5)
            
            # 內積
            valid = np.isfinite(d2e.values) & np.isfinite(d2i.values)
            if np.sum(valid) == 0:
                results.append({"StockId": stock_id, "Date": date, "raw": 0.0})
                continue
            
            dot = np.mean(d2e.values[valid] * d2i.values[valid])
            
            if not np.isfinite(dot):
                results.append({"StockId": stock_id, "Date": date, "raw": 0.0})
                continue
            
            asym = np.sign(prices[-1] - prices[0])
            results.append({"StockId": stock_id, "Date": date, "raw": dot * asym})
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        result_df[self.name] = result_df.groupby("StockId")["raw"].transform(
            lambda x: zscore_rolling(x, window=20, eps=1e-10)
        )
        result_df[self.name] = safe_clip_fillna(result_df[self.name])
        
        return result_df[["StockId", "Date", self.name]]
