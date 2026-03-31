import pandas as pd
import numpy as np
from typing import Union, Tuple

def preprocess_single_stock_broker(data: pd.DataFrame, return_broker_day: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    df = data.copy()
    
    # 1. Base Metrics
    df["net_buy"] = df["BuyQtm"] - df["SellQtm"]
    df["total_vol"] = df["BuyQtm"] + df["SellQtm"]

    df["BuyQtm_clipped"] = df["BuyQtm"].clip(lower=0)
    df["SellQtm_clipped"] = df["SellQtm"].clip(lower=0)
    df["Price_x_Buy"] = df["Price"] * df["BuyQtm_clipped"]
    df["Price_x_Sell"] = df["Price"] * df["SellQtm_clipped"]
    df["_has_buy"] = (df["BuyQtm"] > 0).astype(int)

    # 2. Broker-Day Aggregation
    broker_day = df.groupby(["StockId", "Date", "BrokerId"], as_index=False).agg(
        BuyQtm=("BuyQtm", "sum"),
        SellQtm=("SellQtm", "sum"),
        SumPriceBuy=("Price_x_Buy", "sum"),
        SumPriceSell=("Price_x_Sell", "sum"),
        SumBuyClipped=("BuyQtm_clipped", "sum"),
        SumSellClipped=("SellQtm_clipped", "sum"),
        BuyEntryCount=("_has_buy", "sum"),
    )
    
    broker_day["NetBuy"] = broker_day["BuyQtm"] - broker_day["SellQtm"]
    broker_day["TotalVol"] = broker_day["BuyQtm"] + broker_day["SellQtm"]
    broker_day["AvgBuyPrice"] = broker_day["SumPriceBuy"] / (broker_day["SumBuyClipped"] + 1e-8)

    # Save broker_day before it gets enriched with DailyVWAP
    broker_day_original = broker_day.copy()

    # 3. Daily VWAP
    daily_wt = broker_day.groupby(["StockId", "Date"], as_index=False).agg(
        pw=("SumPriceBuy", "sum"),
        pw2=("SumPriceSell", "sum"),
        w=("SumBuyClipped", "sum"),
        w2=("SumSellClipped", "sum"),
    )
    daily_wt["DailyVWAP"] = (daily_wt["pw"] + daily_wt["pw2"]) / (daily_wt["w"] + daily_wt["w2"] + 1e-8)
    broker_day = broker_day.merge(daily_wt[["StockId", "Date", "DailyVWAP"]], on=["StockId", "Date"], how="left")

    # 4. Daily Aggregation (Vectorized)
    
    # Flow Divergence & Activity
    grp_df = df.groupby(["StockId", "Date"])
    net_buy_df = df["net_buy"]
    
    n_buy = grp_df.apply(lambda g: (g["net_buy"] > 0).sum())
    n_sell = grp_df.apply(lambda g: (g["net_buy"] < 0).sum())
    total_brokers = n_buy + n_sell
    div = np.where(total_brokers > 0, n_buy / total_brokers, 0.5)
    
    total_net = grp_df["net_buy"].sum()
    
    # Entropy (Fully Vectorized)
    sum_vol = grp_df["total_vol"].transform("sum")
    p = np.where(sum_vol > 0, df["total_vol"] / sum_vol, 0)
    p_log = np.where(p > 0, p * np.log(p), 0)
    entropy_df = df.copy()
    entropy_df["p_log"] = p_log
    entropy = -entropy_df.groupby(["StockId", "Date"])["p_log"].sum()
    
    # Top 5 Consistency Setup (Fully Vectorized)
    df_sorted = df.sort_values(by=["StockId", "Date", "net_buy"], ascending=[True, True, False])
    top5_df = df_sorted.groupby(["StockId", "Date"]).head(5)
    top5_sets = top5_df.groupby(["StockId", "Date"])["BrokerId"].apply(set)
    
    # Build daily dataframe
    daily = pd.DataFrame({
        "raw_flow_diverg": div,
        "raw_broker_entropy": entropy,
        "_total_net": total_net,
        "_n_active": total_brokers,
        "_top5": top5_sets,
        "DailyVWAP": daily_wt.set_index(["StockId", "Date"])["DailyVWAP"]
    })

    # 5. 0223 features on pre-aggregated
    bd_grp = broker_day.groupby(["StockId", "Date"])
    total_buy_bd = bd_grp["BuyQtm"].sum()
    total_sell_bd = bd_grp["SellQtm"].sum()
    
    buy_sq = broker_day.copy()
    buy_sq["buy_ratio_sq"] = (buy_sq["BuyQtm"] / buy_sq.groupby(["StockId", "Date"])["BuyQtm"].transform("sum")) ** 2
    buy_hhi = buy_sq.groupby(["StockId", "Date"])["buy_ratio_sq"].sum()
    buy_hhi[total_buy_bd == 0] = 0.0
    
    sell_sq = broker_day.copy()
    sell_sq["sell_ratio_sq"] = (sell_sq["SellQtm"] / sell_sq.groupby(["StockId", "Date"])["SellQtm"].transform("sum")) ** 2
    sell_hhi = sell_sq.groupby(["StockId", "Date"])["sell_ratio_sq"].sum()
    sell_hhi[total_sell_bd == 0] = 0.0
    
    hhi_diff = buy_hhi - sell_hhi
    active_brokers_bd = bd_grp.size()
    
    dt_vol = broker_day[["BuyQtm", "SellQtm"]].min(axis=1)
    broker_day["dt_vol"] = dt_vol
    daytrade_vol = broker_day.groupby(["StockId", "Date"])["dt_vol"].sum()
    dt_intensity = daytrade_vol / (total_buy_bd + total_sell_bd)
    dt_intensity[(total_buy_bd + total_sell_bd) == 0] = 0.0
    
    daily["raw_hhi_diff"] = hhi_diff.fillna(0.0)
    daily["raw_hhi_buy"] = buy_hhi.fillna(0.0)
    daily["raw_hhi_sell"] = sell_hhi.fillna(0.0)
    daily["raw_retail_part"] = active_brokers_bd.fillna(0.0)
    daily["raw_dt_int"] = dt_intensity.fillna(0.0)

    # 6. Time Series Windowing (Vectorized across stocks)
    daily = daily.reset_index()
    daily = daily.sort_values(["StockId", "Date"])
    
    # Consistency
    daily["raw_top5_consistency"] = np.nan
    for stock_id, grp in daily.groupby("StockId"):
        top5_list = grp["_top5"].tolist()
        consist_vals = []
        for i in range(len(top5_list)):
            if i < 5:
                consist_vals.append(np.nan)
                continue
            curr = top5_list[i]
            if not curr:
                consist_vals.append(np.nan)
                continue
                
            overlaps = []
            for j in range(1, 6):
                prev = top5_list[i - j]
                if prev:
                    ov = len(curr & prev) / max(len(curr), 1)
                    overlaps.append(ov)
                    
            val = np.mean(overlaps) if overlaps else 0.0
            consist_vals.append(val)
        daily.loc[grp.index, "raw_top5_consistency"] = consist_vals
        
    # Persistence
    pivot = df.pivot_table(index=["StockId", "Date"], columns="BrokerId", values="net_buy", fill_value=0)
    sign = np.sign(pivot)
    
    roll_sum = sign.groupby(level="StockId").rolling(5, min_periods=5).sum().reset_index(level=0, drop=True)
    pb = (roll_sum == 5).sum(axis=1)
    ps = (roll_sum == -5).sum(axis=1)
    active = (sign != 0).sum(axis=1)
    persist = (pb - ps) / (active + 1e-9)
    
    # Align persist back to daily
    daily = daily.set_index(["StockId", "Date"])
    daily["raw_broker_persist"] = persist.reindex(daily.index).fillna(0.0)

    # Net Reversal
    daily["_net_ma5"] = daily.groupby("StockId")["_total_net"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    daily["_net_std20"] = daily.groupby("StockId")["_total_net"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    daily["raw_net_reversal"] = (daily["_total_net"] - daily["_net_ma5"]) / (daily["_net_std20"] + 1e-9)

    # Activity Surprise
    daily["_active_ma20"] = daily.groupby("StockId")["_n_active"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    daily["raw_activity_surprise"] = daily["_n_active"] / (daily["_active_ma20"] + 1e-9)
    
    # 7. Script B Features (Vectorized)
    day_vols = bd_grp["TotalVol"].sum()
    
    # B2 & B3 Concentration
    top_buy = broker_day.sort_values(["BuyQtm", "BuyEntryCount", "BrokerId"], ascending=[False, False, True]).groupby(["StockId", "Date"]).head(5)
    top_buy_sum = top_buy.groupby(["StockId", "Date"])["BuyQtm"].sum()
    daily["_conviction_buyer_ratio"] = (top_buy_sum / total_buy_bd).fillna(0.0)
    
    top_sell = broker_day.sort_values("SellQtm", ascending=False).groupby(["StockId", "Date"]).head(5)
    top_sell_sum = top_sell.groupby(["StockId", "Date"])["SellQtm"].sum()
    daily["_conviction_seller_ratio"] = (top_sell_sum / total_sell_bd).fillna(0.0)
    
    # B6 Newcomer
    top_net = broker_day.sort_values("NetBuy", ascending=False).groupby(["StockId", "Date"]).head(10)
    top_net_sum = top_net.groupby(["StockId", "Date"])["NetBuy"].sum()
    daily["_top10_net_buy_concentration"] = (top_net_sum / day_vols).fillna(0.0)
    
    # B7 Loyal
    block_mask = broker_day["NetBuy"] > (broker_day.groupby(["StockId", "Date"])["TotalVol"].transform("sum") * 0.1)
    block_sum = broker_day[block_mask].groupby(["StockId", "Date"])["NetBuy"].sum()
    daily["_loyal_inflow"] = (block_sum / (day_vols + 1e-8)).fillna(0.0)
    
    # B8 Polarization
    broker_day["net_ratio"] = broker_day["NetBuy"] / (broker_day["TotalVol"] + 1e-8)
    def _std_if_enough(x):
        return np.std(x) if len(x) >= 3 else 0.0
    polarization = broker_day.groupby(["StockId", "Date"])["net_ratio"].apply(_std_if_enough)
    daily["_polarization_index"] = polarization.fillna(0.0)
    
    # B1 Active broker count (NOT HHI - the name was misleading)
    daily["_broker_count"] = bd_grp.size().astype(float)
    
    # C2 Small Broker Net Momentum
    small_thresh = broker_day.groupby(["StockId", "Date"])["TotalVol"].transform(lambda x: x.quantile(0.5))
    small_brokers = broker_day[broker_day["TotalVol"] <= small_thresh]
    
    small_buy_vwap_num = small_brokers.groupby(["StockId", "Date"])["SumPriceBuy"].sum()
    small_buy_vwap_den = small_brokers.groupby(["StockId", "Date"])["SumBuyClipped"].sum()
    small_buy_vwap = small_buy_vwap_num / (small_buy_vwap_den + 1e-8)
    
    daily_vwap_s = daily_wt.set_index(["StockId", "Date"])["DailyVWAP"]
    
    C2 = (small_buy_vwap / daily_vwap_s) - 1.0
    
    # === Alpha v2 Custom Aggregations ===
    # F5: Bot 80% (Retail) Buyer VWAP
    bot80_thresh = broker_day.groupby(["StockId", "Date"])["TotalVol"].transform(lambda x: x.quantile(0.8))
    bot80_brokers = broker_day[(broker_day["TotalVol"] <= bot80_thresh) & (broker_day["BuyQtm"] > broker_day["SellQtm"])]
    bot80_buy_vwap_num = bot80_brokers.groupby(["StockId", "Date"])["SumPriceBuy"].sum()
    bot80_buy_vwap_den = bot80_brokers.groupby(["StockId", "Date"])["SumBuyClipped"].sum()
    bot80_buy_vwap = bot80_buy_vwap_num / (bot80_buy_vwap_den + 1e-8)
    daily["raw_bot80_buy_vwap"] = bot80_buy_vwap.reindex(daily.index).values

    # F9: Top 10% (Smart Money) Passive Buyer VWAP (Buy > Sell)
    # The original for Top10 uses NetBuy > 0 inside get_top10_buyer_vwap? Let me check. We'll wait.
    net_buyers = broker_day[broker_day["BuyQtm"] > broker_day["SellQtm"]]
    top10_thresh = net_buyers.groupby(["StockId", "Date"])["TotalVol"].transform(lambda x: x.quantile(0.9))
    top10_brokers = net_buyers[net_buyers["TotalVol"] >= top10_thresh]
    top10_buy_vwap_num = top10_brokers.groupby(["StockId", "Date"])["SumPriceBuy"].sum()
    top10_buy_vwap_den = top10_brokers.groupby(["StockId", "Date"])["SumBuyClipped"].sum()
    top10_buy_vwap = top10_buy_vwap_num / (top10_buy_vwap_den + 1e-8)
    daily["raw_top10_buy_vwap"] = top10_buy_vwap.reindex(daily.index).values
    # === End Alpha v2 Custom ===

    # Clean output masking
    valid_mask = (bd_grp.size() >= 3) & (day_vols > 0)
    daily["_newcomer_net_momentum"] = np.where((small_brokers.groupby(["StockId", "Date"]).size() > 0) & (daily_vwap_s > 0), C2, 0.0)
    
    # === Add New Broker Features (Raw) ===

    # F1: f_broker_concentration_shift (Raw Daily HHI of Broker Vol)
    # HHI = sum(share_i^2) where share_i = broker_vol / total_vol_that_day
    b_vol_sq = broker_day.copy()
    b_vol_sq["share_sq"] = (b_vol_sq["TotalVol"] / b_vol_sq.groupby(["StockId", "Date"])["TotalVol"].transform("sum")) ** 2
    daily["raw_broker_hhi"] = b_vol_sq.groupby(["StockId", "Date"])["share_sq"].sum().reindex(daily.index).values

    # F2: f_net_buy_persistence_slope (Raw Daily NetBuy sum)
    daily_net_buy = broker_day.groupby(["StockId", "Date"])["NetBuy"].sum()
    daily["raw_daily_net_buy"] = daily_net_buy.reindex(daily.index).values

    # F3: f_broker_herding_intensity (Raw Day Herding)
    def _herding(g):
        active = len(g)
        if active == 0: return np.nan
        p = (g["NetBuy"] > 0).sum() / active
        return abs(2 * p - 1)
    herding_raw = broker_day.groupby(["StockId", "Date"]).apply(_herding)
    daily["raw_broker_herding"] = herding_raw.reindex(daily.index).values

    # F4: f_consensus_fracture (Raw Ratio)
    # Requires past 10 days NetBuy per broker
    pivot = broker_day.pivot_table(index=["StockId", "Date"], columns="BrokerId", values="NetBuy", fill_value=0)
    
    past10_net = pivot.groupby(level="StockId").rolling(10, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
    
    fracture_vals = []
    daily_indices = daily.index
    for idx in daily_indices:
        if idx not in pivot.index:
            fracture_vals.append(np.nan)
            continue
            
        today_row = pivot.loc[idx]
        active_brokers = today_row[today_row != 0].index
        if len(active_brokers) == 0:
            fracture_vals.append(np.nan)
            continue
            
        if idx in past10_net.index:
            past_row = past10_net.loc[idx].reindex(active_brokers)
        else:
            fracture_vals.append(np.nan)
            continue
            
        today_net = today_row.reindex(active_brokers)
        has_history = past_row.notna()
        
        same_dir = ((past_row > 0) & (today_net > 0)) | ((past_row < 0) & (today_net < 0))
        opp_dir = ((past_row > 0) & (today_net < 0)) | ((past_row < 0) & (today_net > 0))
        
        trend_vol = today_net[has_history & same_dir].abs().sum()
        reversal_vol = today_net[has_history & opp_dir].abs().sum()
        
        max_vol = max(trend_vol, reversal_vol)
        min_vol = min(trend_vol, reversal_vol)
        val = min_vol / (max_vol + 1e-10) if max_vol > 0 else 0.0
        fracture_vals.append(val)
        
    daily["raw_consensus_fracture"] = fracture_vals

    # === Alpha v4 Physics Features ===
    # 1. f_mac_field_current_divergence
    p_net = df["net_buy"].clip(lower=0)
    p_net_sum = p_net.groupby([df["StockId"], df["Date"]]).transform("sum")
    df["_p_net_sq"] = np.where(p_net_sum > 0, (p_net / p_net_sum)**2, 0)
    daily["raw_mac_field_current_divergence"] = df.groupby(["StockId", "Date"])["_p_net_sq"].sum().reindex(daily.index).values
    df.drop(columns=["_p_net_sq"], inplace=True)

    # 2. f_mac_broker_connectivity_breakdown
    df = df.sort_values(["StockId", "Date", "BrokerId"])
    top10_vol = df.sort_values("total_vol", ascending=False, kind="mergesort").groupby(["StockId", "Date"]).head(10)
    top10_vol["_is_one_way"] = (top10_vol["BuyQtm"] == top10_vol["total_vol"]) | (top10_vol["SellQtm"] == top10_vol["total_vol"])
    conn_b = top10_vol.groupby(["StockId", "Date"])["_is_one_way"].mean()
    daily["raw_mac_broker_connectivity_breakdown"] = conn_b.reindex(daily.index).fillna(0.0).values
    # === End Alpha v4 Physics ===

    # === Alpha v5 QIT Physics Features (5) ===
    
    # 1. f_qit_topological_volume_defects
    broker_day["NetAmt"] = broker_day["SumPriceBuy"] - broker_day["SumPriceSell"]
    is_net_buyer = broker_day["NetBuy"] > 0
    broker_day["buyer_vwap"] = np.where(is_net_buyer, broker_day["NetAmt"] / broker_day["NetBuy"].replace(0, 1), np.inf)
    
    daily_buy_amt = broker_day.groupby(["StockId", "Date"])["SumPriceBuy"].transform("sum")
    daily_buy_vol = broker_day.groupby(["StockId", "Date"])["SumBuyClipped"].transform("sum")
    vwap_defect = np.where(daily_buy_vol > 0, daily_buy_amt / daily_buy_vol, 0.0)
    
    broker_day["is_defect"] = is_net_buyer & (broker_day["buyer_vwap"] < vwap_defect)
    broker_day["defect_vol"] = np.where(broker_day["is_defect"], broker_day["TotalVol"], 0)
    defect_sum = broker_day.groupby(["StockId", "Date"])["defect_vol"].sum()
    daily["raw_qit_topological_volume_defects"] = np.where(day_vols > 0, defect_sum.reindex(daily.index).fillna(0) / day_vols.reindex(daily.index).replace(0, 1), 0.0)

    # 2. f_qit_liquidity_catalysis_ratio
    mean_vol = bd_grp["TotalVol"].mean()
    broker_day["_mean_vol"] = broker_day.set_index(["StockId", "Date"]).index.map(mean_vol)
    broker_day["is_catalyst"] = (broker_day["NetBuy"].abs() / broker_day["TotalVol"].replace(0, 1) < 0.05) & (broker_day["TotalVol"] > broker_day["_mean_vol"])
    broker_day["cat_vol"] = np.where(broker_day["is_catalyst"], broker_day["TotalVol"], 0)
    cat_sum = broker_day.groupby(["StockId", "Date"])["cat_vol"].sum()
    daily["raw_qit_liquidity_catalysis_ratio"] = np.where(day_vols > 0, cat_sum.reindex(daily.index).fillna(0) / day_vols.reindex(daily.index).replace(0, 1), 0.0)

    # 3. f_qit_information_causality_bound
    max_p = df.groupby(["StockId", "Date"])["Price"].max()
    min_p = df.groupby(["StockId", "Date"])["Price"].min()
    amp = np.where(min_p > 0, (max_p - min_p) / min_p, 0.0)
    broker_count = daily["_n_active"]
    daily["raw_qit_information_causality_bound"] = np.where(broker_count > 0, pd.Series(amp, index=min_p.index).reindex(daily.index).fillna(0) / broker_count, 0.0)

    # 4. f_qit_dilution_factor_inverse
    broker_day["share_sq"] = (broker_day["TotalVol"] / broker_day.groupby(["StockId", "Date"])["TotalVol"].transform("sum")) ** 2
    daily["raw_qit_dilution_factor_inverse"] = broker_day.groupby(["StockId", "Date"])["share_sq"].sum().reindex(daily.index).values

    # 5. f_qit_athermal_free_energy
    from scipy.stats import entropy
    net_buyers = broker_day[broker_day["NetBuy"] > 0]
    def _entropy_net(g):
        n = len(g)
        if n <= 1: return 0.0
        p = g["NetBuy"] / g["NetBuy"].sum()
        q = np.ones(n) / n
        return float(entropy(p.values, q))
    daily["raw_qit_athermal_free_energy"] = net_buyers.groupby(["StockId", "Date"]).apply(_entropy_net).reindex(daily.index).fillna(0.0).values
    
    # === End Alpha v5 QIT Physics Features ===

    # === Alpha v6: New Broker Feature 20260312 ===
    # f_net_antiferro_broker_interaction: Top5-buy vs Top5-sell concentration × direction
    _af_eps = 1e-10
    def _antiferro(g):
        if len(g) < 10:
            return np.nan
        g_sorted = g.sort_values("NetBuy", ascending=False)
        top5_buy  = g_sorted.head(5)
        top5_sell = g_sorted.tail(5)
        total_vol = g["TotalVol"].sum()
        buy_conc  = top5_buy["BuyQtm"].sum() / (total_vol + _af_eps)
        sell_conc = top5_sell["SellQtm"].sum() / (total_vol + _af_eps)
        buy_net   = top5_buy["NetBuy"].sum()
        sell_net  = top5_sell["NetBuy"].sum()
        return float((buy_conc - sell_conc) * (buy_net / (abs(buy_net) + abs(sell_net) + _af_eps)))
    _af_res = broker_day.groupby(["StockId", "Date"]).apply(_antiferro)
    daily["raw_net_antiferro_broker_interaction"] = _af_res.reindex(daily.index).values
    # === End Alpha v6 ===

    # === Alpha v12: 生態模擬特徵新增 Raw Columns (20260318) ===
    
    # 1. f_allee_reproduction_deficit: Top10 Buy vs Bottom80% Sell
    # Top 10 brokers by buy volume
    broker_day_sorted = broker_day.sort_values(["BuyQtm", "BuyEntryCount", "BrokerId"], ascending=[False, False, True])
    top10_buy_df = broker_day_sorted.groupby(["StockId", "Date"]).head(10)
    top10_buy_agg = top10_buy_df.groupby(["StockId", "Date"])["BuyQtm"].sum()
    daily["raw_top10_buy"] = top10_buy_agg.reindex(daily.index).fillna(0).values
    
    # Bottom 80% brokers by sell volume
    broker_day["SellPctRank"] = broker_day.groupby(["StockId", "Date"])["SellQtm"].rank(pct=True, ascending=True)
    bot80_sell_df = broker_day[broker_day["SellPctRank"] <= 0.8]
    bot80_sell_agg = bot80_sell_df.groupby(["StockId", "Date"])["SellQtm"].sum()
    daily["raw_bot80_sell"] = bot80_sell_agg.reindex(daily.index).fillna(0).values
    
    # 2. f_core_source_contribution_ratio: VWAP below/above breakdown
    broker_day["IsBelowVWAP"] = broker_day["AvgBuyPrice"] <= broker_day["DailyVWAP"]
    broker_day["LowVWAPBuy"] = np.where(broker_day["IsBelowVWAP"], broker_day["BuyQtm"], 0)
    broker_day["HighVWAPBuy"] = np.where(~broker_day["IsBelowVWAP"], broker_day["BuyQtm"], 0)
    
    # Top 10 brokers by total buy (re-sort after adding columns)
    broker_day_sorted = broker_day.sort_values(["BuyQtm", "BuyEntryCount", "BrokerId"], ascending=[False, False, True])
    top10_by_buy = broker_day_sorted.groupby(["StockId", "Date"]).head(10)
    top10_vwap_below_agg = top10_by_buy.groupby(["StockId", "Date"])["LowVWAPBuy"].sum()
    top10_vwap_above_agg = top10_by_buy.groupby(["StockId", "Date"])["HighVWAPBuy"].sum()
    daily["raw_top10_low_buy"] = top10_vwap_below_agg.reindex(daily.index).fillna(0).values
    daily["raw_top10_high_buy"] = top10_vwap_above_agg.reindex(daily.index).fillna(0).values
    
    # 3. f_cumulative_dose_threshold: Consecutive buying >= 3 days
    # This requires tracking consecutive buy days per broker - compute raw here
    broker_pivot = broker_day.pivot_table(index=["StockId", "Date"], columns="BrokerId", values="NetBuy", fill_value=0)
    is_buying = (broker_pivot > 0).astype(int)
    
    # Use rolling(3) to check 3+ consecutive buying days per broker (data-length-independent)
    # rolling(3).sum()==3 is equivalent to consecutive>=3, but only needs 3 rows of lookback
    is_infected = is_buying.groupby(level="StockId", group_keys=False).apply(
        lambda g: g.rolling(3, min_periods=3).sum() == 3
    )
    infected_net_buy = (broker_pivot * is_infected).sum(axis=1)
    daily["raw_infected_net_buy"] = infected_net_buy.reindex(daily.index).fillna(0).values
    
    # 4. f_dynamic_r0_momentum: Top 10 net buy (already mostly computed, add raw)
    top10_net_buy_agg = top10_by_buy.groupby(["StockId", "Date"])["NetBuy"].sum()
    daily["raw_top10_net_buy"] = top10_net_buy_agg.reindex(daily.index).fillna(0).values
    
    # 5. f_herd_immunity_barrier: Retail (Bottom 80%) VWAP with 20-day rolling
    # Already have retail VWAP calculation, add rolling
    retail_thresh = broker_day.groupby(["StockId", "Date"])["TotalVol"].transform(lambda x: x.quantile(0.8))
    retail_df = broker_day[broker_day["TotalVol"] > retail_thresh]
    retail_vwap_num = retail_df.groupby(["StockId", "Date"])["SumPriceBuy"].sum()
    retail_vwap_den = retail_df.groupby(["StockId", "Date"])["SumBuyClipped"].sum()
    retail_vwap = (retail_vwap_num / (retail_vwap_den + 1e-8)).reindex(daily.index)
    daily["raw_retail_vwap"] = retail_vwap.fillna(0).values
    
    # 6. f_inbreeding_depression_risk: Top 5 Jaccard similarity
    # Calculate Top 5 brokers set per day
    top5_by_buy = broker_day.sort_values(["BuyQtm", "BuyEntryCount", "BrokerId"], ascending=[False, False, True]).groupby(["StockId", "Date"]).head(5)
    top5_sets = top5_by_buy.groupby(["StockId", "Date"])["BrokerId"].apply(set)
    
    # Calculate 5-day Jaccard
    jaccard_scores = []
    for idx in daily.index:
        if idx not in top5_sets.index:
            jaccard_scores.append(np.nan)
            continue
        current_set = top5_sets.loc[idx]
        # Get previous 4 days
        prev_idx = daily.index.get_loc(idx)
        if prev_idx < 4:
            jaccard_scores.append(np.nan)
            continue
        history_sets = set()
        for j in range(max(0, prev_idx - 4), prev_idx):
            prev_idx_loc = daily.index[j]
            if prev_idx_loc in top5_sets.index:
                history_sets.update(top5_sets.loc[prev_idx_loc])
        intersection = len(history_sets.intersection(current_set))
        union = len(history_sets.union(current_set))
        jacc = intersection / union if union > 0 else 0
        jaccard_scores.append(jacc)
    daily["raw_top5_jaccard"] = jaccard_scores
    
    # 7. f_intraguild_predation_skew: Top5 NetBuy vs Mid10 (6-15) NetSell
    _buy_rank_sorted = broker_day.sort_values(["StockId", "Date", "BuyQtm", "BuyEntryCount", "BrokerId"], ascending=[True, True, False, False, True])
    broker_day["BuyRankByBuy"] = _buy_rank_sorted.groupby(["StockId", "Date"]).cumcount() + 1
    top5_net_df = broker_day[(broker_day["BuyRankByBuy"] <= 5) & (broker_day["NetBuy"] > 0)]
    top5_net_agg = top5_net_df.groupby(["StockId", "Date"])["NetBuy"].sum()
    daily["raw_top5_net_buy"] = top5_net_agg.reindex(daily.index).fillna(0).values
    
    mid10_df = broker_day[(broker_day["BuyRankByBuy"] > 5) & (broker_day["BuyRankByBuy"] <= 15) & (broker_day["NetBuy"] < 0)]
    mid10_net_agg = mid10_df.groupby(["StockId", "Date"])["NetBuy"].sum().abs()
    daily["raw_mid10_net_sell"] = mid10_net_agg.reindex(daily.index).fillna(0).values
    
    # 8. f_primary_secondary_infection_div: Top5 ratio vs Rest participation
    total_buy_bd = broker_day.groupby(["StockId", "Date"])["BuyQtm"].sum()
    top5_ratio = (top5_net_agg / (total_buy_bd + 1e-8)).reindex(daily.index).fillna(0)
    daily["raw_top5_ratio"] = top5_ratio.values
    
    # Rest broker participation (brokers with BuyRank > 5 that actually bought)
    rest_buyers = broker_day[broker_day["BuyRankByBuy"] > 5]
    rest_buy_count = rest_buyers[rest_buyers["BuyQtm"] > 0].groupby(["StockId", "Date"])["BrokerId"].count()
    total_broker_count = broker_day.groupby(["StockId", "Date"]).size()
    rest_participation = (rest_buy_count / (total_broker_count + 1e-8)).reindex(daily.index).fillna(0)
    daily["raw_rest_participation"] = rest_participation.values
    
    # 9. f_resistance_gamma_heterogeneity: Sell HHI / Seller count
    # Already have sell HHI calculation in earlier section
    # Get seller count
    sellers = broker_day[broker_day["SellQtm"] > 0]
    seller_count = sellers.groupby(["StockId", "Date"]).size()
    daily["raw_seller_count"] = seller_count.reindex(daily.index).fillna(0).values
    
    # 10. f_stoichiometric_nutrient_imbalance: Top10 NetValue vs Bot80 NetSell
    # Top 10 net value (only positive net)
    top10_net_val_df = broker_day[(broker_day["BuyRankByBuy"] <= 10)]
    top10_net_value = top10_net_val_df[top10_net_val_df["NetBuy"] > 0].groupby(["StockId", "Date"])["NetBuy"].sum()  # Using NetBuy as proxy for NetValue
    daily["raw_top10_net_value"] = top10_net_value.reindex(daily.index).fillna(0).values
    
    # Bottom 80% net sell (NetQtm < 0)
    bot80_net_sell = broker_day[broker_day["SellPctRank"] <= 0.8]
    bot80_net_sell_agg = bot80_net_sell[bot80_net_sell["NetBuy"] < 0].groupby(["StockId", "Date"])["NetBuy"].sum().abs()
    daily["raw_bot80_net_sell_qtm"] = bot80_net_sell_agg.reindex(daily.index).fillna(0).values
    
    # 11. f_tumor_fingering_breakout: Top1 VWAP vs Rest VWAP
    # Top 1 broker by buy
    top1_df = broker_day.sort_values(["BuyQtm", "BuyEntryCount", "BrokerId"], ascending=[False, False, True]).groupby(["StockId", "Date"]).head(1)
    top1_vwap = top1_df.set_index(["StockId", "Date"])["AvgBuyPrice"]
    daily["raw_top1_vwap"] = top1_vwap.reindex(daily.index).fillna(0).values
    
    # Rest brokers VWAP
    rest_broker_vwap_num = broker_day.groupby(["StockId", "Date"])["SumPriceBuy"].sum()
    rest_broker_vwap_den = broker_day.groupby(["StockId", "Date"])["SumBuyClipped"].sum()
    rest_vwap = (rest_broker_vwap_den + 1e-8)
    # Need to exclude Top1 - recalculate
    broker_day["IsTop1"] = broker_day.sort_values(["BuyQtm", "BuyEntryCount", "BrokerId"], ascending=[False, False, True]).groupby(["StockId", "Date"]).cumcount() == 0
    rest_vwap_agg = broker_day[~broker_day["IsTop1"]].groupby(["StockId", "Date"]).agg(
        RestSumPriceBuy=("SumPriceBuy", "sum"),
        RestSumBuyClipped=("SumBuyClipped", "sum")
    )
    rest_vwap_val = (rest_vwap_agg["RestSumPriceBuy"] / (rest_vwap_agg["RestSumBuyClipped"] + 1e-8)).reindex(daily.index).fillna(0)
    daily["raw_rest_vwap"] = rest_vwap_val.values
    
    # 12. f_tumor_fingering_instability: Pioneer brokers (who traded at daily max price)
    daily_max_price = broker_day.groupby(["StockId", "Date"])["AvgBuyPrice"].max()
    daily["raw_daily_max_price"] = daily_max_price.reindex(daily.index).fillna(0).values
    daily["raw_daily_vwap"] = daily_wt.set_index(["StockId", "Date"])["DailyVWAP"].reindex(daily.index).fillna(0).values
    
    # Pioneer: brokers whose max price equals daily max
    broker_day["MaxBrokerPrice"] = broker_day.groupby(["StockId", "Date"])["AvgBuyPrice"].transform("max")
    broker_day["IsPioneer"] = broker_day["AvgBuyPrice"] == broker_day["MaxBrokerPrice"]
    pioneer_buy = broker_day[broker_day["IsPioneer"]].groupby(["StockId", "Date"])["BuyQtm"].sum()
    daily["raw_pioneer_buy"] = pioneer_buy.reindex(daily.index).fillna(0).values
    
    # Daily total buy for pioneer ratio
    daily_total_buy = broker_day.groupby(["StockId", "Date"])["BuyQtm"].sum()
    daily["raw_daily_total_buy"] = daily_total_buy.reindex(daily.index).fillna(0).values
    
    # 13. f_post_latency_infectivity_jump: Top5 NetBuy + Price Range + Vol Jump
    # Already have top5_net_buy
    # Need daily price range
    daily_high = broker_day.groupby(["StockId", "Date"])["AvgBuyPrice"].max()
    daily_low = broker_day.groupby(["StockId", "Date"])["AvgBuyPrice"].min()
    daily["raw_price_range_5d"] = ((daily_high / (daily_low + 1e-8)) - 1).reindex(daily.index).fillna(0).values
    
    # Volume jump
    vol_ts = broker_day.groupby(["StockId", "Date"])["TotalVol"].sum()
    vol_ma5 = vol_ts.rolling(5, min_periods=1).mean()
    vol_jump = (vol_ts / (vol_ma5.shift(1) + 1e-8)).reindex(daily.index).fillna(1)
    daily["raw_vol_jump"] = vol_jump.values
    
    # Sum5 Top5 Buy (rolling 5)
    top5_net_buy_ts = daily["raw_top5_net_buy"]
    sum5_top5_buy = top5_net_buy_ts.rolling(5, min_periods=1).sum()
    daily["raw_sum5_top5_buy"] = sum5_top5_buy.values
    
    # === End Alpha v12 ===
    
    # === Alpha v17 Game Theory: Focal Point Preprocessing ===
    # raw_is_focal: Flag per-row if Price is a multiple of 10
    # This is used for f_gt_payoff_dominant_focal_point
    df["raw_is_focal"] = (df["Price"] % 10 == 0).astype(int)
    
    # At focal prices: distinct broker count and total buy volume per day
    focal_df = df[df["raw_is_focal"] == 1]
    focal_daily = focal_df.groupby(["StockId", "Date"]).agg(
        raw_focal_distinct_brokers=("BrokerId", "nunique"),
        raw_focal_buy_vol=("BuyQtm", "sum")
    ).reset_index()
    
    # Merge back to daily
    daily = daily.merge(focal_daily, on=["StockId", "Date"], how="left")
    daily["raw_focal_distinct_brokers"] = daily["raw_focal_distinct_brokers"].fillna(0)
    daily["raw_focal_buy_vol"] = daily["raw_focal_buy_vol"].fillna(0)
    
    # === End Alpha v17 Focal Point ===
    
    # Ensure missing valid mask rows are null internally
    clean_cols = ["_conviction_buyer_ratio", "_conviction_seller_ratio", "_top10_net_buy_concentration", "_loyal_inflow", "_polarization_index", "_newcomer_net_momentum"]
    valid_mask_values = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
    for c in clean_cols:
        if c in daily.columns:
            daily.loc[~valid_mask_values, c] = np.nan
        
    if return_broker_day:
        return daily.reset_index(), broker_day_original
    return daily.reset_index()
