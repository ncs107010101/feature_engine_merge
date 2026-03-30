import numpy as np
import pandas as pd
from ...core import BaseFeature, register_feature


@register_feature
class FeatureGtCommitmentDeviceLock(BaseFeature):
    name = "f_gt_commitment_device_lock"
    description = "承諾機制鎖碼：近期大漲(5日>5%)且過去20日累積買超最大之前10大券商今日賣出量為0或極低。ewm(3)標準化。"
    required_columns = ["StockId", "Date", "收盤價"]
    data_combination = "cross_broker_daily"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if "_broker_day" not in kwargs:
            raise ValueError("f_gt_commitment_device_lock requires broker_day data in kwargs")
        
        broker_day = kwargs["_broker_day"].copy()
        broker_day["Date"] = broker_day["Date"].astype(int)
        
        df = data[self.required_columns].copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        broker_pivot = broker_day.pivot_table(
            index="Date", columns="BrokerId", values="BuyQtm", aggfunc="sum", fill_value=0
        ).sort_index()
        sell_pivot = broker_day.pivot_table(
            index="Date", columns="BrokerId", values="SellQtm", aggfunc="sum", fill_value=0
        ).sort_index()
        
        buy_roll20 = broker_pivot.rolling(20, min_periods=1).sum()
        
        all_results = []
        for date in buy_roll20.index:
            row_buy = buy_roll20.loc[date]
            top10_brokers = row_buy.nlargest(10).index.tolist()
            if date in sell_pivot.index:
                today_sell = sell_pivot.loc[date, top10_brokers].sum()
            else:
                today_sell = 0.0
            all_results.append({"Date": date, "Top10_Today_Sell": float(today_sell)})
        
        broker_agg = pd.DataFrame(all_results)
        df = df.merge(broker_agg, on="Date", how="left")
        df["Top10_Today_Sell"] = df["Top10_Today_Sell"].fillna(0)
        
        ret5 = df.groupby("StockId")["收盤價"].pct_change(5)
        is_strong_up = (ret5 > 0.05).astype(float)
        
        sell_q5 = df.groupby("StockId")["Top10_Today_Sell"].transform(
            lambda x: x.rolling(20, min_periods=1).quantile(0.05)
        ).shift(1)
        is_minimal_sell = (df["Top10_Today_Sell"] <= sell_q5.clip(lower=0)).astype(float)
        
        raw = is_strong_up * is_minimal_sell
        out = raw.groupby(df["StockId"]).transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
