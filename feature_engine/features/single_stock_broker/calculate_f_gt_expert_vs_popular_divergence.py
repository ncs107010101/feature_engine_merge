import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FGtExpertVsPopularDivergence(BaseFeature):
    name = "f_gt_expert_vs_popular_divergence"
    description = "Game Theory v17: Popular brokers (high 5d volume) bullish but Expert brokers (60d net accumulators) bearish. Positive → Popular bullish but experts exit → Extreme LOW return"
    required_columns = []  # Uses broker_day from kwargs, not data columns
    data_combination = "single_stock_broker"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Get broker_day from kwargs (set by preprocessing)
        broker_day = kwargs.get('_broker_day')
        if broker_day is None:
            raise ValueError("f_gt_expert_vs_popular_divergence requires broker_day data from preprocessing")
        
        df = broker_day.copy()
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        results = []
        for stock_id, stock_df in df.groupby("StockId", group_keys=False):
            dates = sorted(stock_df["Date"].unique())
            if len(dates) == 0:
                continue
            
            # Build pivot tables
            net_pivot = stock_df.pivot_table(index="Date", columns="BrokerId", values="NetBuy", fill_value=0).sort_index()
            vol_pivot = stock_df.pivot_table(index="Date", columns="BrokerId", values="TotalVol", fill_value=0).sort_index()
            
            # Rolling sums
            vol_roll5 = vol_pivot.rolling(5, min_periods=1).sum()
            net_roll60 = net_pivot.rolling(60, min_periods=1).sum()
            
            daily_vals = []
            for date in vol_roll5.index:
                if date not in vol_roll5.index:
                    daily_vals.append(0.0)
                    continue
                
                # Popular: top 3 by 5-day volume
                pop_brokers = vol_roll5.loc[date].nlargest(3).index.tolist()
                # Expert: top 3 by 60-day net (exclude popular)
                exp_candidates = net_roll60.loc[date].drop(labels=pop_brokers, errors="ignore")
                exp_brokers = exp_candidates.nlargest(3).index.tolist()
                
                pop_net = net_pivot.loc[date, pop_brokers].sum() if date in net_pivot.index else 0
                exp_net = net_pivot.loc[date, exp_brokers].sum() if (date in net_pivot.index and exp_brokers) else 0
                
                is_diverge = int(pop_net > 0 and exp_net < 0)
                magnitude = abs(pop_net - exp_net)
                daily_vals.append(is_diverge * magnitude)
            
            stock_result = pd.DataFrame({
                "Date": dates,
                self.name: daily_vals,
                "StockId": stock_id
            })
            results.append(stock_result)
        
        if not results:
            return pd.DataFrame(columns=["StockId", "Date", self.name])
        
        result_df = pd.concat(results, ignore_index=True)
        result_df = result_df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # Apply zscore_rolling and clip
        out = result_df.groupby("StockId")[self.name].transform(lambda x: zscore_rolling(x, 20))
        result_df[self.name] = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        return result_df[["StockId", "Date", self.name]]
