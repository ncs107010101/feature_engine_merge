import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

# Constants for new features
EPS = 1e-9
NET_BINS = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
N_BINS = 5
SEGMENT_SECS = 30 * 60
MARKET_START = 9 * 3600
N_SEGS = 8
WINDOW = 20
WIN_SHORT = 5
WIN_LONG = 60


def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def kl_divergence(p, q):
    p = np.array(p, dtype=float) + EPS
    q = np.array(q, dtype=float) + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def directed_info_1markov(b_seq, y_seq):
    b = np.array(b_seq, dtype=int)
    y = np.array(y_seq, dtype=int)
    T = len(b)
    if T < 5:
        return 0.0

    counts_full = np.zeros((4, 2), dtype=float)
    counts_y1 = np.zeros((2, 2), dtype=float)

    for t in range(1, T):
        ctx = b[t - 1] * 2 + y[t - 1]
        counts_full[ctx, y[t]] += 1
        counts_y1[y[t - 1], y[t]] += 1

    counts_full += 1e-3
    counts_y1 += 1e-3

    p_full = counts_full / counts_full.sum(axis=1, keepdims=True)
    p_y1 = counts_y1 / counts_y1.sum(axis=1, keepdims=True)

    di = 0.0
    for t in range(1, T):
        ctx = b[t - 1] * 2 + y[t - 1]
        yt = y[t]
        yt1 = y[t - 1]
        p_cond_full = p_full[ctx, yt]
        p_cond_base = p_y1[yt1, yt]
        di += np.log(p_cond_full / p_cond_base)

    return float(max(0.0, di / (T - 1)))

def calculate_price_time_probability(
    tick_df: pd.DataFrame, 
    time_col: str = "Date", 
    price_col: str = "DealPrice",
    cond_mask: pd.Series = None
) -> pd.DataFrame:
    """
    Compute price-time probability: V_condition(Price) / V_total(Price)
    """
    if cond_mask is None or cond_mask.empty:
        return pd.DataFrame(columns=[time_col, price_col, "P_time"])
    
    # Total volume per price
    total_vol = tick_df.groupby([time_col, price_col])["DealCount"].sum().reset_index(name="TotalVol")
    
    # Conditional volume per price
    cond_vol = tick_df[cond_mask].groupby([time_col, price_col])["DealCount"].sum().reset_index(name="CondVol")
    
    prob_df = pd.merge(total_vol, cond_vol, on=[time_col, price_col], how="left")
    prob_df["CondVol"] = prob_df["CondVol"].fillna(0)
    prob_df["P_time"] = prob_df["CondVol"] / (prob_df["TotalVol"] + 1e-8)
    
    return prob_df[[time_col, price_col, "P_time"]]


def preprocess_cross_broker_tick(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = data.copy()

    if "tick_data" not in kwargs:
        raise ValueError("hf_profile requires raw tick data passed as `tick_data` in kwargs.")

    df_tick = kwargs["tick_data"].copy()
    req_tick = ["StockId", "Date", "DealTimeSecond", "DealPrice", "DealCount", "PrFlag"]
    missing_tick = [c for c in req_tick if c not in df_tick.columns]
    if missing_tick:
        raise ValueError(f"tick_data is missing required columns: {missing_tick}")

    df["Date"] = df["Date"].astype(int)
    df_tick["Date"] = df_tick["Date"].astype(int)

    # ------ TICK AGGREGATION ------
    df_tick["ActiveBuy"] = np.where(df_tick["PrFlag"] == 1, df_tick["DealCount"], 0)
    df_tick["ActiveSell"] = np.where(df_tick["PrFlag"] == 0, df_tick["DealCount"], 0)

    # Spread on downticks for f_acellular_gap_formation
    df_tick["IsDowntick"] = (df_tick["PrFlag"] == 0)
    df_tick["PrevDealPr"] = df_tick.groupby(["StockId", "Date"])["DealPrice"].shift(1)
    df_tick["SpreadDown"] = np.where(
        df_tick["IsDowntick"] & (df_tick["PrevDealPr"] > 0),
        (df_tick["DealPrice"] - df_tick["PrevDealPr"]).abs() / df_tick["PrevDealPr"],
        np.nan
    )

    # 5-minute window for refuge/burst identification
    df_tick["MinIdx5"] = df_tick["DealTimeSecond"] // 300
    # PrevDealPr already computed above, reuse it
    df_tick["PriceRange5m"] = np.where(
        df_tick["PrevDealPr"] > 0,
        (df_tick["DealPrice"] - df_tick["PrevDealPr"]).abs() / df_tick["PrevDealPr"],
        0
    )

    # Per 5-minute bar aggregation
    min5_bars = df_tick.groupby(["StockId", "Date", "MinIdx5"]).agg(
        Min5Vol=("DealCount", "sum"),
        Min5ActiveBuy=("ActiveBuy", "sum"),
        Min5Range=("PriceRange5m", "max"),
    ).reset_index()

    # Identify burst minute (max active buy per day)
    burst_min_idx = min5_bars.groupby(["StockId", "Date"])["Min5ActiveBuy"].idxmax()
    burst_minutes_df = min5_bars.loc[burst_min_idx, ["StockId", "Date", "MinIdx5"]].copy()
    burst_minutes_df["is_burst"] = True
    
    # Ensure consistent dtypes for merge
    df_tick["MinIdx5"] = df_tick["MinIdx5"].astype("int64")
    burst_minutes_df["MinIdx5"] = burst_minutes_df["MinIdx5"].astype("int64")

    df_tick = df_tick.merge(
        burst_minutes_df[["StockId", "Date", "MinIdx5", "is_burst"]], 
        on=["StockId", "Date", "MinIdx5"], 
        how="left"
    )
    df_tick["is_burst"] = df_tick["is_burst"].fillna(False)

    # Compute daily mean vol/range for refuge identification
    daily_min5_stats = min5_bars.groupby(["StockId", "Date"]).agg(
        Mean5mVol=("Min5Vol", "mean"),
        Mean5mRange=("Min5Range", "mean"),
    ).reset_index()

    min5_bars = min5_bars.merge(daily_min5_stats, on=["StockId", "Date"], how="left")

    # Refuge: low vol AND low range
    min5_bars["is_refuge_candidate"] = (
        (min5_bars["Min5Vol"] < min5_bars["Mean5mVol"]) & 
        (min5_bars["Min5Range"] < min5_bars["Mean5mRange"])
    )

    refuge_minutes_df = min5_bars[min5_bars["is_refuge_candidate"]][["StockId", "Date", "MinIdx5"]].drop_duplicates()
    refuge_minutes_df = refuge_minutes_df.copy()
    refuge_minutes_df["is_refuge"] = True
    refuge_minutes_df["MinIdx5"] = refuge_minutes_df["MinIdx5"].astype("int64")

    df_tick = df_tick.merge(
        refuge_minutes_df[["StockId", "Date", "MinIdx5", "is_refuge"]], 
        on=["StockId", "Date", "MinIdx5"], 
        how="left"
    )
    df_tick["is_refuge"] = df_tick["is_refuge"].fillna(False)

    # Late-day quiet period: 13:00+ with low volatility
    df_tick["IsLateDay"] = (df_tick["DealTimeSecond"] >= 13 * 3600)

    # Merge daily mean range to late ticks
    min5_bars_for_late = min5_bars[["StockId", "Date", "MinIdx5", "Min5Range", "Mean5mRange"]].copy()
    min5_bars_for_late["MinIdx5"] = min5_bars_for_late["MinIdx5"].astype("int64")
    df_tick = df_tick.merge(min5_bars_for_late, on=["StockId", "Date", "MinIdx5"], how="left")
    df_tick["Mean5mRange"] = df_tick["Mean5mRange"].fillna(0)

    # Late-day quiet: late day AND range < daily mean
    df_tick["is_late_quiet"] = df_tick["IsLateDay"] & (df_tick["Min5Range"] < df_tick["Mean5mRange"])

    # Compute P_burst, P_refuge, P_late_quiet per price level
    prob_burst = calculate_price_time_probability(df_tick, cond_mask=df_tick["is_burst"])
    prob_burst.columns = ["Date", "DealPrice", "P_burst"]

    prob_refuge = calculate_price_time_probability(df_tick, cond_mask=df_tick["is_refuge"])
    prob_refuge.columns = ["Date", "DealPrice", "P_refuge"]

    prob_late_quiet = calculate_price_time_probability(df_tick, cond_mask=df_tick["is_late_quiet"])
    prob_late_quiet.columns = ["Date", "DealPrice", "P_late_quiet"]

    df_tick["IsMorning"] = (df_tick["DealTimeSecond"] <= 9 * 3600 + 30 * 60)
    df_tick["IsLateDayExcl"] = (df_tick["DealTimeSecond"] >= 13 * 3600) & (df_tick["DealTimeSecond"] < 13 * 3600 + 25 * 60)

    df_tick["MorningVol"] = df_tick["DealCount"] * df_tick["IsMorning"]
    df_tick["MorningActiveBuy"] = df_tick["ActiveBuy"] * df_tick["IsMorning"]
    df_tick["LateDayExclActiveBuy"] = df_tick["ActiveBuy"] * df_tick["IsLateDayExcl"]

    tick_agg = df_tick.groupby(["StockId", "Date", "DealPrice"]).agg(
        TickVol=("DealCount", "sum"),
        TickActiveBuy=("ActiveBuy", "sum"),
        TickActiveSell=("ActiveSell", "sum"),
        TickSpreadDown=("SpreadDown", "mean"),
        MorningVol=("MorningVol", "sum"),
        MorningActiveBuy=("MorningActiveBuy", "sum"),
        LateDayExclActiveBuy=("LateDayExclActiveBuy", "sum"),
    ).reset_index()

    tick_agg["DealValue"] = tick_agg["DealPrice"] * tick_agg["TickVol"]
    
    # ---- NEW: Tick daily aggregation for F26-F40 intermediates ----
    daily_stats = tick_agg.groupby(["StockId", "Date"]).agg(
        DailyHigh=("DealPrice", "max"),
        DailyLow=("DealPrice", "min"),
        TotalValue=("DealValue", "sum"),
        TotalVol=("TickVol", "sum"),
    ).reset_index()
    daily_stats["DailyVWAP"] = daily_stats["TotalValue"] / daily_stats["TotalVol"].replace(0, 1)
    
    tick_day_base = df_tick.groupby(["StockId", "Date"]).agg(
        raw_open_p=("DealPrice", "first"),
        raw_close_p=("DealPrice", "last"),
        raw_buy_vol=("ActiveBuy", "sum"),
        raw_sell_vol=("ActiveSell", "sum"),
        raw_total_vol_tick=("DealCount", "sum"),
    ).reset_index()
    
    # Compute return and delta_p
    tick_day_base["raw_delta_p"] = tick_day_base["raw_close_p"] - tick_day_base["raw_open_p"]
    tick_day_base["raw_tick_ret"] = tick_day_base["raw_delta_p"] / (tick_day_base["raw_open_p"] + EPS)
    
    # Large/small tick segmentation using q85 (per NEW feature code)
    q85 = df_tick.groupby(["StockId", "Date"])["DealCount"].transform(lambda x: x.quantile(0.85))
    df_tick["_is_large"] = (df_tick["DealCount"] >= q85).astype(int)
    df_tick["_is_small"] = (df_tick["DealCount"] < q85).astype(int)
    df_tick["_large_buy"] = df_tick["ActiveBuy"] * df_tick["_is_large"]
    df_tick["_small_buy"] = df_tick["ActiveBuy"] * df_tick["_is_small"]
    df_tick["_large_sell"] = df_tick["ActiveSell"] * df_tick["_is_large"]
    df_tick["_small_sell"] = df_tick["ActiveSell"] * df_tick["_is_small"]
    
    tick_size_agg = df_tick.groupby(["StockId", "Date"]).agg(
        raw_large_buy=("_large_buy", "sum"),
        raw_small_buy=("_small_buy", "sum"),
        raw_large_sell=("_large_sell", "sum"),
        raw_small_sell=("_small_sell", "sum"),
    ).reset_index()
    
    tick_day_base = tick_day_base.merge(tick_size_agg, on=["StockId", "Date"], how="left")
    
    # Spread and price acceleration
    tick_day_base["raw_spread_mean"] = df_tick.groupby(["StockId", "Date"]).apply(
        lambda g: (g["SellPr"] - g["BuyPr"]).mean()
    ).reset_index(drop=True)
    
    # Price 2nd derivative (acceleration) - use existing order (sorted by TotalQty)
    def compute_price_accel(g):
        dp = g["DealPrice"].values
        delta_p = np.diff(dp, prepend=dp[0])
        accel = np.diff(delta_p, prepend=delta_p[0])
        return np.mean(accel)
    tick_accel = df_tick.groupby(["StockId", "Date"]).apply(compute_price_accel).reset_index(name="raw_a_price")
    tick_day_base = tick_day_base.merge(tick_accel, on=["StockId", "Date"], how="left")
    tick_day_base["raw_a_price"] = tick_day_base["raw_a_price"].fillna(0)
    
    # Fill NaN for new columns
    for col in ["raw_large_buy", "raw_small_buy", "raw_large_sell", "raw_small_sell", "raw_spread_mean"]:
        if col in tick_day_base.columns:
            tick_day_base[col] = tick_day_base[col].fillna(0)

    tick_agg = pd.merge(
        tick_agg,
        daily_stats[["StockId", "Date", "DailyHigh", "DailyLow", "DailyVWAP"]],
        on=["StockId", "Date"],
    )

    tick_agg["PriceRange"] = tick_agg["DailyHigh"] - tick_agg["DailyLow"]
    tick_agg["IsTop20"] = (tick_agg["DealPrice"] >= (tick_agg["DailyLow"] + 0.8 * tick_agg["PriceRange"])).astype(int)
    tick_agg["IsBot20"] = (tick_agg["DealPrice"] <= (tick_agg["DailyLow"] + 0.2 * tick_agg["PriceRange"])).astype(int)
    tick_agg["IsBlwVWAP"] = (tick_agg["DealPrice"] < tick_agg["DailyVWAP"]).astype(int)

    base_vol = tick_agg["TickActiveBuy"] + tick_agg["TickActiveSell"]
    tick_agg["ActiveBuyRatio"] = np.where(base_vol > 0, tick_agg["TickActiveBuy"] / base_vol, 0.5)
    tick_agg["ActiveSellRatio"] = np.where(base_vol > 0, tick_agg["TickActiveSell"] / base_vol, 0.5)
    tick_agg["MorningRatio"] = np.where(tick_agg["TickVol"] > 0, tick_agg["MorningVol"] / tick_agg["TickVol"], 0)
    tick_agg["MorningABuyRatio"] = np.where(tick_agg["TickVol"] > 0, tick_agg["MorningActiveBuy"] / tick_agg["TickVol"], 0)
    tick_agg["LateDayExclABuyRatio"] = np.where(tick_agg["TickVol"] > 0, tick_agg["LateDayExclActiveBuy"] / tick_agg["TickVol"], 0)

    # ------ MERGE WITH RAW BROKER DATA ------
    merged = pd.merge(
        df, tick_agg,
        left_on=["StockId", "Date", "Price"],
        right_on=["StockId", "Date", "DealPrice"],
        how="inner",
    )

    merged = merged.merge(prob_burst, on=["Date", "DealPrice"], how="left")
    merged = merged.merge(prob_refuge, on=["Date", "DealPrice"], how="left")
    merged = merged.merge(prob_late_quiet, on=["Date", "DealPrice"], how="left")

    merged["P_burst"] = merged["P_burst"].fillna(0)
    merged["P_refuge"] = merged["P_refuge"].fillna(0)
    merged["P_late_quiet"] = merged["P_late_quiet"].fillna(0)

    merged["EstBurstBuy"] = merged["BuyQtm"] * merged["P_burst"]
    merged["EstRefugeBuy"] = merged["BuyQtm"] * merged["P_refuge"]
    merged["EstLateQuietBuy"] = merged["BuyQtm"] * merged["P_late_quiet"]

    merged["EstActiveBuy"] = merged["BuyQtm"] * merged["ActiveBuyRatio"]
    merged["EstPassiveBuy"] = merged["BuyQtm"] * merged["ActiveSellRatio"]
    merged["EstActiveSell"] = merged["SellQtm"] * merged["ActiveSellRatio"]
    merged["EstPassiveSell"] = merged["SellQtm"] * merged["ActiveBuyRatio"]

    merged["EstMorningBuy"] = merged["BuyQtm"] * merged["MorningRatio"]
    merged["EstMorningSell"] = merged["SellQtm"] * merged["MorningRatio"]
    merged["EstMorningABuy"] = merged["BuyQtm"] * merged["MorningABuyRatio"]
    merged["EstLateDayExclABuy"] = merged["BuyQtm"] * merged["LateDayExclABuyRatio"]

    merged["EstABuy_Top20"] = merged["EstActiveBuy"] * merged["IsTop20"]
    merged["EstABuy_Bot20"] = merged["EstActiveBuy"] * merged["IsBot20"]
    merged["EstPBuy_BlwVWAP"] = merged["EstPassiveBuy"] * merged["IsBlwVWAP"]

    daily_broker = merged.groupby(["StockId", "Date", "BrokerId"]).agg(
        TotalBuy=("BuyQtm", "sum"),
        TotalSell=("SellQtm", "sum"),
        ActiveBuy=("EstActiveBuy", "sum"),
        PassiveBuy=("EstPassiveBuy", "sum"),
        ActiveSell=("EstActiveSell", "sum"),
        PassiveSell=("EstPassiveSell", "sum"),
        MorningBuy=("EstMorningBuy", "sum"),
        MorningSell=("EstMorningSell", "sum"),
        MorningABuy=("EstMorningABuy", "sum"),
        LateDayExclABuy=("EstLateDayExclABuy", "sum"),
        ABuy_Top20=("EstABuy_Top20", "sum"),
        ABuy_Bot20=("EstABuy_Bot20", "sum"),
        PBuy_BlwVWAP=("EstPBuy_BlwVWAP", "sum"),
        EstBurstBuy=("EstBurstBuy", "sum"),
        EstRefugeBuy=("EstRefugeBuy", "sum"),
        EstLateQuietBuy=("EstLateQuietBuy", "sum"),
    ).reset_index()

    daily_broker["TotalVol"] = daily_broker["TotalBuy"] + daily_broker["TotalSell"]

    daily_broker["BrokerSizeRank"] = daily_broker.groupby(["StockId", "Date"])["TotalVol"].rank(
        pct=True, method="first"
    )

    # Pre-calculate HHI components
    daily_broker["ABuyPct"] = daily_broker["ActiveBuy"] / daily_broker.groupby(["StockId", "Date"])["ActiveBuy"].transform("sum").replace(0, 1)
    daily_broker["ASellPct"] = daily_broker["ActiveSell"] / daily_broker.groupby(["StockId", "Date"])["ActiveSell"].transform("sum").replace(0, 1)
    daily_broker["PBuyPct"] = daily_broker["PassiveBuy"] / daily_broker.groupby(["StockId", "Date"])["PassiveBuy"].transform("sum").replace(0, 1)
    
    daily_broker["HHI_Buy_Sq"] = daily_broker["ABuyPct"] ** 2
    daily_broker["HHI_Sell_Sq"] = daily_broker["ASellPct"] ** 2
    daily_broker["HHI_PBuy_Sq"] = daily_broker["PBuyPct"] ** 2

    # Pre-calculate masked components
    mask_bot80 = daily_broker["BrokerSizeRank"] < 0.80
    mask_top20 = daily_broker["BrokerSizeRank"] >= 0.80
    mask_top10 = daily_broker["BrokerSizeRank"] >= 0.90
    mask_top5 = daily_broker["BrokerSizeRank"] >= 0.95

    daily_broker["top10_PBuy_BlwVWAP"] = np.where(mask_top10, daily_broker["PBuy_BlwVWAP"], 0)
    daily_broker["bot80_ABuy_Top20"] = np.where(mask_bot80, daily_broker["ABuy_Top20"], 0)
    daily_broker["bot80_ABuy_Bot20"] = np.where(mask_bot80, daily_broker["ABuy_Bot20"], 0)
    daily_broker["top20_LateDayExclABuy"] = np.where(mask_top20, daily_broker["LateDayExclABuy"], 0)
    daily_broker["top10_ABuy_Top20"] = np.where(mask_top10, daily_broker["ABuy_Top20"], 0)
    
    daily_broker["bot80_MorningSell"] = np.where(mask_bot80, daily_broker["MorningSell"], 0)
    daily_broker["top10_MorningBuy"] = np.where(mask_top10, daily_broker["MorningBuy"], 0)
    daily_broker["top5_PassiveSell"] = np.where(mask_top5, daily_broker["PassiveSell"], 0)
    
    daily_broker["top10_MorningABuy"] = np.where(mask_top10, daily_broker["MorningABuy"], 0)
    daily_broker["top10_PassiveSell"] = np.where(mask_top10, daily_broker["PassiveSell"], 0)
    daily_broker["top10_ActiveBuy"] = np.where(mask_top10, daily_broker["ActiveBuy"], 0)
    daily_broker["top10_PassiveBuy"] = np.where(mask_top10, daily_broker["PassiveBuy"], 0)
    # NetSell rank (by NetSell value directly, for Top5_NetSell calculation)
    daily_broker["NetSell"] = daily_broker["TotalSell"] - daily_broker["TotalBuy"]
    daily_broker["NetSellRank"] = daily_broker.groupby(["StockId", "Date"])["NetSell"].rank(
        method="first", ascending=False
    )
    mask_top5_netsell = daily_broker["NetSellRank"] <= 5
    
    daily_broker["top10_NetSell"] = np.where(mask_top10, daily_broker["TotalSell"] - daily_broker["TotalBuy"], 0.0)
    daily_broker["Top5_NetSell"] = np.where(mask_top5_netsell, daily_broker["NetSell"], 0.0)
    daily_broker["top5_Sell"] = np.where(mask_top5, daily_broker["TotalSell"], 0.0)

    # Perform a single groupby sum for all components
    cols_to_sum = [
        "TotalVol", "PassiveSell", "PassiveBuy",
        "top10_PBuy_BlwVWAP", "bot80_ABuy_Top20", "bot80_ABuy_Bot20", "top20_LateDayExclABuy", "top10_ABuy_Top20",
        "bot80_MorningSell", "top10_MorningBuy", "top5_PassiveSell",
        "HHI_Buy_Sq", "HHI_Sell_Sq", "HHI_PBuy_Sq",
        "top10_MorningABuy", "top10_PassiveSell", "top10_ActiveBuy", "top10_PassiveBuy",
        "top10_NetSell", "Top5_NetSell", "top5_Sell",
        "EstBurstBuy", "EstRefugeBuy", "EstLateQuietBuy",
    ]
    
    stock_date = daily_broker.groupby(["StockId", "Date"])[cols_to_sum].sum().reset_index()

    features = stock_date[["StockId", "Date"]].copy()

    vol_div = stock_date["TotalVol"] + 1
    psell_div = stock_date["PassiveSell"] + 1e-5

    # Features 1-5
    features["r_smart_accumulation_blwVWAP"] = stock_date["top10_PBuy_BlwVWAP"] / vol_div
    features["r_retail_trapped_top20"] = stock_date["bot80_ABuy_Top20"] / vol_div
    features["r_retail_falling_knife_bot20"] = stock_date["bot80_ABuy_Bot20"] / vol_div
    features["r_lateday_conviction_excl"] = stock_date["top20_LateDayExclABuy"] / vol_div
    features["r_smart_breakout_top20"] = stock_date["top10_ABuy_Top20"] / vol_div

    # Features 6-8
    features["r_morning_retail_sel_bot_80_pct"] = stock_date["bot80_MorningSell"] / vol_div
    features["r_morning_smart_buy_top_10_pct"] = stock_date["top10_MorningBuy"] / vol_div
    features["r_large_psell_wall_top_5_pct_conc"] = stock_date["top5_PassiveSell"] / psell_div

    # Feature 9
    features["r_aggressor_concentration"] = stock_date["HHI_Buy_Sq"] - stock_date["HHI_Sell_Sq"]

    # Features 10-14
    features["f_morning_smart_active_buy"] = stock_date["top10_MorningABuy"] / vol_div
    features["f_morning_retail_panic"] = stock_date["bot80_MorningSell"] / vol_div
    features["f_large_passive_sell_wall"] = stock_date["top10_PassiveSell"] / vol_div
    features["f_large_broker_active_buy"] = stock_date["top10_ActiveBuy"] / vol_div
    features["f_large_broker_passive_buying_pressure"] = stock_date["top10_PassiveBuy"] / vol_div

    # Feature 15
    features["f_concentrated_passive_buy"] = stock_date["HHI_PBuy_Sq"]

    # === Ecological Feature Intermediates ===
    # raw_top10_net_sell_ratio for f_acellular_gap_formation
    features["raw_top10_net_sell_ratio"] = stock_date["top10_NetSell"] / (stock_date["TotalVol"] + 1)

    # raw_passive_buy_vol for f_predator_driven_extinction_rate
    features["raw_passive_buy_vol"] = stock_date["PassiveBuy"]

    # raw_top5_sell_vol for f_predator_driven_extinction_rate
    features["raw_top5_sell_vol"] = stock_date["top5_Sell"]

    # === Alpha v3: New Cross Broker Tick Feature ===
    # raw_broker_sync_intensity (sync rate of top-5 net-buying brokers' execution times)
    
    # 1. Build tick price -> time mapping (Date, Price -> set of seconds)
    tick_subset = df_tick[["Date", "DealPrice", "DealTimeSecond"]].copy()
    tick_time_map = tick_subset.groupby(["Date", "DealPrice"])["DealTimeSecond"].apply(lambda x: frozenset(x.unique())).to_dict()
    
    # 2. Build broker price sets (Date, BrokerId -> set of prices)
    broker_prices = df.groupby(["Date", "BrokerId"])["Price"].apply(lambda x: frozenset(x.unique())).to_dict()
    
    # 3. Top 5 NetBuy brokers per day using RAW broker data
    raw_bd = df.groupby(["StockId", "Date", "BrokerId"]).agg(
        TotalBuy=("BuyQtm", "sum"),
        TotalSell=("SellQtm", "sum"),
    ).reset_index()
    raw_bd["NetBuy"] = raw_bd["TotalBuy"] - raw_bd["TotalSell"]
    
    # NB_BuyBrokers: count of brokers with NetBuy > 0 per stock-date (for F36)
    nb_buy_brokers = raw_bd[raw_bd["NetBuy"] > 0].groupby(["StockId", "Date"]).size().reset_index(name="NB_BuyBrokers")
    
    # ---- NEW: Broker NetBuy intermediates for F26-F40 ----
    # Sort by TotalBuy descending, then by NetBuy descending as tiebreaker for deterministic Top10 selection
    raw_bd = raw_bd.sort_values(["StockId", "Date", "TotalBuy", "NetBuy"], ascending=[True, True, False, False])
    raw_bd["BuyRank"] = raw_bd.groupby(["StockId", "Date"]).cumcount()
    mask_top10_nb = raw_bd["BuyRank"] < 10
    mask_top5_nb = raw_bd["BuyRank"] < 5
    mask_retail_nb = raw_bd["BuyRank"] >= 10
    raw_bd["top10_net_buy"] = np.where(mask_top10_nb, raw_bd["NetBuy"], 0.0)
    raw_bd["top5_net_buy"] = np.where(mask_top5_nb, raw_bd["NetBuy"], 0.0)
    raw_bd["retail_net_buy"] = np.where(mask_retail_nb, raw_bd["NetBuy"], 0.0)
    broker_netbuy_agg = raw_bd.groupby(["StockId", "Date"]).agg(
        raw_top10_net_buy=("top10_net_buy", "sum"),
        raw_top5_net_buy=("top5_net_buy", "sum"),
        raw_retail_net_buy=("retail_net_buy", "sum"),
        raw_total_buy_broker=("TotalBuy", "sum"),
    ).reset_index()
    # ---- END NEW ----
    
    sync_records = []
    dates = features["Date"].unique()
    
    from itertools import combinations
    for d in dates:
        day_bd = raw_bd[raw_bd["Date"] == d]
        if len(day_bd) < 5:
            sync_records.append({"Date": d, "raw_broker_sync_intensity": np.nan})
            continue
            
        top5_ids = day_bd.nlargest(5, "NetBuy")["BrokerId"].values
        
        # Find active times for top 5
        broker_times = {}
        for bid in top5_ids:
            key = (d, bid)
            prices = broker_prices.get(key, frozenset())
            times = set()
            for p in prices:
                tk = (d, p)
                if tk in tick_time_map:
                    times.update(tick_time_map[tk])
            broker_times[bid] = times
            
        # Pairwise Jaccard
        pairs = list(combinations(top5_ids, 2))
        if not pairs:
            sync_records.append({"Date": d, "raw_broker_sync_intensity": np.nan})
            continue
            
        j_sum, v_pairs = 0.0, 0
        for a, b in pairs:
            sa = set(broker_times.get(a, set()))
            sb = set(broker_times.get(b, set()))
            union_len = len(sa | sb)
            if union_len > 0:
                j_sum += len(sa & sb) / union_len
                v_pairs += 1
                
        val = j_sum / v_pairs if v_pairs > 0 else np.nan
        sync_records.append({"Date": d, "raw_broker_sync_intensity": val})
        
    sync_df = pd.DataFrame(sync_records)
    features = pd.merge(features, sync_df, on="Date", how="left")

    # === Alpha v4 Physics Features ===
    # 1. f_eth_conservative_work_divergence
    df_tick["MinIdx"] = df_tick["DealTimeSecond"] // 60
    min_bars = df_tick.groupby(["Date", "MinIdx"], as_index=False).agg(last_pr=("DealPrice", "last"))
    min_bars["prev_pr"] = min_bars.groupby("Date")["last_pr"].shift(1)
    min_bars["ret_abs"] = np.where(min_bars["prev_pr"] > 0, (min_bars["last_pr"] - min_bars["prev_pr"]).abs() / min_bars["prev_pr"], 0)
    path_len = min_bars.groupby("Date")["ret_abs"].sum()
    
    brk_sum = raw_bd.copy() # raw_bd has StockId, Date, BrokerId, NetBuy
    brk_sum["NetBuy"] = brk_sum["NetBuy"].clip(lower=0)
    top5_buy = brk_sum.sort_values(["Date", "NetBuy"], ascending=[True, False]).groupby("Date").head(5)
    top5_net_buy_sum = top5_buy.groupby("Date")["NetBuy"].sum()
    
    features = features.set_index("Date")
    features["raw_eth_conservative_work_divergence"] = (top5_net_buy_sum.reindex(features.index).fillna(0) / (path_len.reindex(features.index).fillna(0) * 1000.0 + 1e-5)).values
    features = features.reset_index()
    # === End Alpha v4 Physics ===

    # === Alpha v5: f_qm_measurement_incompatibility intermediates (20260312) ===
    # A: tick-level daily active-buy ratio − 0.5  (direction of market aggression)
    tick_day_ab = df_tick.groupby(["StockId", "Date"]).agg(
        _ab=("ActiveBuy", "sum"),
        _as=("ActiveSell", "sum")
    )
    tick_day_ab["_abr"] = (
        tick_day_ab["_ab"] / (tick_day_ab["_ab"] + tick_day_ab["_as"] + 1e-10) - 0.5
    )
    # B: large-broker (top 10% by vol) net flow ratio
    _lb_mask = daily_broker["BrokerSizeRank"] >= 0.90
    daily_broker["_lb_net"]  = np.where(_lb_mask, daily_broker["ActiveBuy"] - daily_broker["ActiveSell"], 0.0)
    daily_broker["_lb_tvol"] = np.where(_lb_mask, daily_broker["TotalVol"], 0.0)
    _lb_agg = daily_broker.groupby(["StockId", "Date"]).agg(
        _lbn=("_lb_net", "sum"), _lbt=("_lb_tvol", "sum"))
    _lb_agg["_lbnr"] = _lb_agg["_lbn"] / (_lb_agg["_lbt"] + 1e-10)

    # Merge both into features
    features = features.set_index(["StockId", "Date"])
    features["raw_qm_tick_active_buy_ratio"] = tick_day_ab["_abr"].reindex(features.index).values
    features["raw_qm_large_broker_net_ratio"] = _lb_agg["_lbnr"].reindex(features.index).values
    features = features.reset_index()
    # === End Alpha v5 ===

    # === Alpha v6: f_gt_information_design_obfuscation intermediate ===
    # Compute daily obfuscation index from tick data: 50-tick bins
    # obfuscation = mean(flip_counts * dc_cvs) per bin
    # When n_ticks < 100, set obfuscation to 0 (matches original implementation)
    BIN_SIZE = 50
    _tick_count = df_tick.groupby(["StockId", "Date"]).size().reset_index(name="_n_ticks")
    _tick_sorted = df_tick.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
    _tick_sorted["_bin_id"] = _tick_sorted.groupby(["StockId", "Date"]).cumcount() // BIN_SIZE

    # Compute flips within each bin - first within group diff
    _tick_sorted["_prev_pf"] = _tick_sorted.groupby(["StockId", "Date", "_bin_id"])["PrFlag"].shift(1)
    _tick_sorted["_is_flip"] = (_tick_sorted["_prev_pf"].notna()) & (_tick_sorted["PrFlag"] != _tick_sorted["_prev_pf"])
    _tick_sorted["_is_flip"] = _tick_sorted["_is_flip"].astype(int)
    
    _bin_stats = _tick_sorted.groupby(["StockId", "Date", "_bin_id"]).agg(
        _flip_cnt=("_is_flip", "sum"),
        _dc_mean=("DealCount", "mean"),
        _dc_std=("DealCount", lambda x: x.std(ddof=0)),
    ).reset_index()
    _bin_stats["_dc_cv"] = _bin_stats["_dc_std"] / (_bin_stats["_dc_mean"] + EPS)

    # Daily obfuscation index
    _obf_bins = _bin_stats.copy()
    _obf_bins["_obf_bin"] = _obf_bins["_flip_cnt"] * _obf_bins["_dc_cv"]
    _obf_daily = _obf_bins.groupby(["StockId", "Date"])["_obf_bin"].mean().reset_index()
    _obf_daily.columns = ["StockId", "Date", "raw_obfuscation_index"]

    # Apply threshold: set to 0 when n_ticks < BIN_SIZE * 2
    _obf_daily = _obf_daily.merge(_tick_count, on=["StockId", "Date"], how="left")
    _obf_daily.loc[_obf_daily["_n_ticks"] < BIN_SIZE * 2, "raw_obfuscation_index"] = 0.0
    _obf_daily = _obf_daily[["StockId", "Date", "raw_obfuscation_index"]]

    features = features.merge(_obf_daily, on=["StockId", "Date"], how="left")
    features["raw_obfuscation_index"] = features["raw_obfuscation_index"].fillna(0)
    # === End Alpha v6 ===

    # Top5_NetSell (intermediate for f_gt_information_design_obfuscation)
    # Market-wide: top 5 brokers across ALL stocks by NetSell per date
    _raw_bd = df.groupby(["Date", "BrokerId"]).agg(
        TotalBuy=("BuyQtm", "sum"),
        TotalSell=("SellQtm", "sum"),
    ).reset_index()
    _raw_bd["NetSell"] = _raw_bd["TotalSell"] - _raw_bd["TotalBuy"]

    _top5_netsell_list = []
    for date, grp in _raw_bd.groupby("Date"):
        top5_sum = grp.nlargest(5, "NetSell")["NetSell"].sum()
        _top5_netsell_list.append({"Date": int(date), "Top5_NetSell": top5_sum})

    _top5_netsell_df = pd.DataFrame(_top5_netsell_list)

    features = features.merge(_top5_netsell_df, on="Date", how="left")
    features["Top5_NetSell"] = features["Top5_NetSell"].fillna(0)

    # === IT Wiretap Channel Features: Intermediate Variables ===
    # For f_it_holevo_privacy_premium: Use actual broker nets and volume-weighted values
    NET_BINS_ARR = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    N_BINS = 5

    df_tick["_is_buy"] = (df_tick["PrFlag"] == 1).astype(int)

    tick_day_stats = df_tick.groupby(["StockId", "Date"]).agg(
        _tick_cnt=("PrFlag", "count"),
        _tick_buy_cnt=("_is_buy", "sum"),
        _total_vol=("DealCount", "sum"),
    ).reset_index()
    tick_day_stats["p_active_up"] = tick_day_stats["_tick_buy_cnt"] / tick_day_stats["_tick_cnt"].clip(lower=1)

    q75 = df_tick.groupby(["StockId", "Date"])["DealCount"].transform(lambda x: x.quantile(0.75))
    q25 = df_tick.groupby(["StockId", "Date"])["DealCount"].transform(lambda x: x.quantile(0.25))
    df_tick["_is_large"] = (df_tick["DealCount"] >= q75).astype(int)
    df_tick["_is_small"] = (df_tick["DealCount"] <= q25).astype(int)

    large_tick = df_tick[df_tick["_is_large"] == 1]
    small_tick = df_tick[df_tick["_is_small"] == 1]

    large_buy = large_tick.groupby(["StockId", "Date"])["_is_buy"].sum()
    large_cnt = large_tick.groupby(["StockId", "Date"])["_is_buy"].count()
    small_buy = small_tick.groupby(["StockId", "Date"])["_is_buy"].sum()
    small_cnt = small_tick.groupby(["StockId", "Date"])["_is_buy"].count()

    tick_day_stats = tick_day_stats.set_index(["StockId", "Date"])
    tick_day_stats["p_large_buy"] = (large_buy / large_cnt.clip(lower=1)).reindex(tick_day_stats.index).fillna(0.5)
    tick_day_stats["p_small_buy"] = (small_buy / small_cnt.clip(lower=1)).reindex(tick_day_stats.index).fillna(0.5)

    broker_stats = df.groupby(["StockId", "Date", "BrokerId"]).agg(
        Buy=("BuyQtm", "sum"),
        Sell=("SellQtm", "sum"),
    ).reset_index()
    broker_stats["Total"] = broker_stats["Buy"] + broker_stats["Sell"]
    broker_stats = broker_stats[broker_stats["Total"] > 0]
    broker_stats["net"] = (broker_stats["Buy"] - broker_stats["Sell"]) / broker_stats["Total"]

    q80 = broker_stats.groupby(["StockId", "Date"])["Total"].transform(lambda x: x.quantile(0.80))
    broker_stats["is_big"] = (broker_stats["Total"] >= q80).astype(int)

    broker_stats["net_bin"] = pd.cut(broker_stats["net"], bins=NET_BINS_ARR, labels=False, include_lowest=True)
    broker_stats["net_bin"] = broker_stats["net_bin"].fillna(2).astype(int)

    big_brokers = broker_stats[broker_stats["is_big"] == 1]
    small_brokers = broker_stats[broker_stats["is_big"] == 0]

    def compute_dist_vectorized(sub_df):
        counts = np.bincount(sub_df["net_bin"].values, minlength=N_BINS).astype(float)
        counts += 1e-3
        return counts / counts.sum()

    dist_big = big_brokers.groupby(["StockId", "Date"]).apply(compute_dist_vectorized)
    dist_small = small_brokers.groupby(["StockId", "Date"]).apply(compute_dist_vectorized)

    big_agg = big_brokers.groupby(["StockId", "Date"]).agg(
        big_net_sum=("net", "sum"),
        big_buy_sum=("Buy", "sum"),
        big_sell_sum=("Sell", "sum"),
    )
    big_agg["big_net_ratio"] = (big_agg["big_buy_sum"] - big_agg["big_sell_sum"]) / (big_agg["big_buy_sum"] + big_agg["big_sell_sum"] + 1e-9)

    small_agg = small_brokers.groupby(["StockId", "Date"]).agg(
        small_net_sum=("net", "sum"),
        small_buy_sum=("Buy", "sum"),
        small_sell_sum=("Sell", "sum"),
    )
    small_agg["small_net_ratio"] = (small_agg["small_buy_sum"] - small_agg["small_sell_sum"]) / (small_agg["small_buy_sum"] + small_agg["small_sell_sum"] + 1e-9)

    # === NEW: Add actual broker weights and nets for Holevo calculation ===
    # This matches new_feature_code's approach: volume-weighted individual broker data
    def get_broker_weights_nets(sub_df):
        """Returns (weights, nets) arrays for big brokers - volume weighted"""
        if len(sub_df) < 2:
            return [1.0], [0.0]
        total_vol = sub_df["Total"].sum()
        weights = (sub_df["Total"] / total_vol).values
        nets = sub_df["net"].values
        return weights.tolist(), nets.tolist()

    broker_data_big = big_brokers.groupby(["StockId", "Date"]).apply(get_broker_weights_nets)
    # broker_data_big is a Series of (weights_list, nets_list) tuples
    
    features = features.set_index(["StockId", "Date"])
    features["raw_p_active_up"] = tick_day_stats["p_active_up"].reindex(features.index).fillna(0.5)
    features["raw_p_large_buy"] = tick_day_stats["p_large_buy"].reindex(features.index).fillna(0.5)
    features["raw_p_small_buy"] = tick_day_stats["p_small_buy"].reindex(features.index).fillna(0.5)

    features["raw_dist_big"] = dist_big.reindex(features.index).apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    features["raw_dist_small"] = dist_small.reindex(features.index).apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    features["raw_big_net_ratio"] = big_agg["big_net_ratio"].reindex(features.index).fillna(0.0)
    features["raw_small_net_ratio"] = small_agg["small_net_ratio"].reindex(features.index).fillna(0.0)
    
    # Store actual broker weights and nets for Holevo calculation
    features["raw_broker_weights_big"] = broker_data_big.reindex(features.index).apply(lambda x: x[0] if isinstance(x, tuple) else [1.0]).tolist()
    features["raw_broker_nets_big"] = broker_data_big.reindex(features.index).apply(lambda x: x[1] if isinstance(x, tuple) else [0.0]).tolist()

    # === Add ewm_net for Information Theory Features ===
    # ewm_net = ewm of net_ratio_big with span=5 (for F08 feature)
    features = features.reset_index()
    features = features.sort_values(["StockId", "Date"])
    features["raw_ewm_net"] = features.groupby("StockId")["raw_big_net_ratio"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )
    features = features.set_index(["StockId", "Date"])
    # === End ewm_net ===

    valid_tick = df_tick[df_tick["RefPr"] > 0]
    if not valid_tick.empty:
        pub_signal = valid_tick.groupby(["StockId", "Date"]).apply(
            lambda x: (x["BuyPr"] > x["RefPr"]).mean()
        )
        features["raw_p_public_buy"] = pub_signal.reindex(features.index).fillna(0.5)
    else:
        features["raw_p_public_buy"] = 0.5

    features = features.reset_index()
    # === End IT Wiretap Channel ===

    # === Add Rolling Skew for F14 ===
    # Rolling skew of net_ratio_big and p_active_up with 20-day window
    features = features.sort_values(["StockId", "Date"])

    # Create rolling skew columns
    features["raw_rolling_skew_net"] = features.groupby("StockId")["raw_big_net_ratio"].transform(
        lambda x: x.rolling(20, min_periods=10).skew()
    )
    features["raw_rolling_skew_tick"] = features.groupby("StockId")["raw_p_active_up"].transform(
        lambda x: x.rolling(20, min_periods=10).skew()
    )

    features = features.set_index(["StockId", "Date"])
    # === End Rolling Skew ===

    # === Add 30-min Tick Segments for F20 ===
    features = features.reset_index()
    features = features.sort_values(["StockId", "Date"])

    def compute_tick_segments_vectorized(df_tick_day):
        if len(df_tick_day) < 20:
            return [{"h": np.nan, "vwap": np.nan}] * N_SEGS
        df_s = df_tick_day.sort_values("DealTimeSecond")
        seg_results = []
        for s in range(N_SEGS):
            s_start = MARKET_START + s * SEGMENT_SECS
            s_end = s_start + SEGMENT_SECS
            seg = df_s[(df_s["DealTimeSecond"] >= s_start) & (df_s["DealTimeSecond"] < s_end)]
            if len(seg) < 3:
                seg_results.append({"h": np.nan, "vwap": np.nan})
                continue
            p_buy = (seg["PrFlag"] == 1).mean()
            h = binary_entropy(p_buy)
            vol = seg["DealCount"].sum()
            vwap = (seg["DealPrice"] * seg["DealCount"]).sum() / (vol + EPS)
            seg_results.append({"h": h, "vwap": vwap})
        return seg_results

    tick_seg_records = []
    for (sid, date), grp in df_tick.groupby(["StockId", "Date"]):
        segs = compute_tick_segments_vectorized(grp)
        tick_seg_records.append({"StockId": sid, "Date": date, "raw_tick_segments": segs})
    tick_seg_df = pd.DataFrame(tick_seg_records)
    features = features.merge(tick_seg_df, on=["StockId", "Date"], how="left")
    # === End Tick Segments ===

    features = features.sort_values(["StockId", "Date"]).reset_index(drop=True)

    WINDOW_DI = 20
    di_series = np.full(len(features), np.nan)
    B_arr = (features["raw_big_net_ratio"] > 0).astype(int).values
    Y_arr = (features["raw_p_active_up"] > 0.5).astype(int).values

    stock_groups = features.groupby("StockId")

    for stock_id, group in stock_groups:
        indices = group.index.tolist()
        B_stock = B_arr[indices]
        Y_stock = Y_arr[indices]
        for i in range(WINDOW_DI, len(indices)):
            win_b = B_stock[i - WINDOW_DI:i]
            win_y = Y_stock[i - WINDOW_DI:i]
            if (win_b.sum() < 2) or (win_b.sum() > WINDOW_DI - 2):
                continue
            di_series[indices[i]] = directed_info_1markov(win_b, win_y)

    features["raw_di_series"] = di_series

    features = features.sort_values(["StockId", "Date"]).reset_index(drop=True)

    dist_hist_60d = [np.nan] * len(features)
    net_ratio_hist_20d = [np.nan] * len(features)
    net_ratio_hist_60d = [np.nan] * len(features)

    stock_groups = features.groupby("StockId")

    for stock_id, group in stock_groups:
        indices = group.index.tolist()
        dist_col = group["raw_dist_big"].values
        net_col = group["raw_big_net_ratio"].values

        for i in range(len(indices)):
            if i < WINDOW:
                continue

            window_dist = dist_col[max(0, i-60):i]
            window_net = net_col[max(0, i-60):i+1]

            valid_dists = [d for d in window_dist if isinstance(d, (list, np.ndarray)) and len(d) == N_BINS]
            if len(valid_dists) >= 30:
                stacked = np.vstack(valid_dists)
                mean_dist = np.nanmean(stacked, axis=0)
                mean_dist = mean_dist / (mean_dist.sum() + EPS) if mean_dist.sum() > 0 else np.ones(N_BINS) / N_BINS
                dist_hist_60d[indices[i]] = list(mean_dist)

            if len(window_net) >= WINDOW + 1:
                net_ratio_hist_20d[indices[i]] = list(window_net[-WINDOW - 1:-1])
            
            if len(window_net) >= 60:
                net_ratio_hist_60d[indices[i]] = list(window_net[-60:])

    features["raw_dist_big_hist_60d"] = dist_hist_60d
    features["raw_net_ratio_hist_20d"] = net_ratio_hist_20d
    features["raw_net_ratio_hist_60d"] = net_ratio_hist_60d

    features = features.sort_values(["StockId", "Date"]).reset_index(drop=True)

    # === Alpha v10 (Group 7): Feedback Channel Features (F23-F25) ===
    # F23: f_it_feedback_capacity_utilization_skew
    # Compute tick-level PrFlag entropy per day
    def compute_feedback_capacity_skew(df_tick_day):
        """Compute CapUtil_up - CapUtil_dn from tick PrFlag sequence"""
        if len(df_tick_day) < 20:
            return np.nan
        
        # Sort by DealTimeSecond (and TotalQty for sub-second precision)
        sorted_df = df_tick_day.sort_values(["DealTimeSecond", "TotalQty"])
        prf = sorted_df["PrFlag"].values
        Y = (prf == 1).astype(int)
        T = len(Y)
        if T < 10:
            return np.nan
        
        p_up = Y.mean()
        p_up = np.clip(p_up, EPS, 1 - EPS)
        H_Y = binary_entropy(p_up)
        
        # Conditional entropies
        y_prev = Y[:-1]
        y_curr = Y[1:]
        
        def cond_entropy_given_prev(cond_val):
            mask = y_prev == cond_val
            sub = y_curr[mask]
            if len(sub) < 2:
                return H_Y
            p = sub.mean()
            return binary_entropy(np.clip(p, EPS, 1 - EPS))
        
        H_Y_given_up = cond_entropy_given_prev(1)
        H_Y_given_dn = cond_entropy_given_prev(0)
        
        # Information retention (entropy reduction / H_Y, normalized)
        cap_util_up = (H_Y - H_Y_given_up) / (H_Y + EPS)
        cap_util_dn = (H_Y - H_Y_given_dn) / (H_Y + EPS)
        
        return float(cap_util_up - cap_util_dn)
    
    # Compute F23 per day
    f23_records = []
    for (sid, date), grp in df_tick.groupby(["StockId", "Date"]):
        f23_val = compute_feedback_capacity_skew(grp)
        f23_records.append({"StockId": sid, "Date": date, "raw_feedback_capacity_skew": f23_val})
    
    f23_df = pd.DataFrame(f23_records)
    features = features.merge(f23_df, on=["StockId", "Date"], how="left")
    # === End F23 ===
    
    # F24: f_it_mi_conservation_imbalance - needs delta_vwap
    # Compute delta_vwap (VWAP change from previous day to current day)
    features = features.sort_values(["StockId", "Date"])
    
    # Get daily VWAP from tick data
    tick_vwap = df_tick.groupby(["StockId", "Date"]).apply(
        lambda g: (g["DealPrice"] * g["DealCount"]).sum() / g["DealCount"].sum()
    ).reset_index(name="tick_vwap")
    
    features = features.merge(tick_vwap, on=["StockId", "Date"], how="left")
    
    # Compute delta_vwap (using previous day's VWAP - avoid lookahead)
    features["raw_tick_vwap"] = features["tick_vwap"]
    features["raw_delta_vwap"] = features.groupby("StockId")["tick_vwap"].pct_change()
    features.drop(columns=["tick_vwap"], inplace=True)
    # === End F24 ===
    
    # F25: f_it_broadcast_channel_entropy_gap - can use existing raw_p_large_buy, raw_p_small_buy
    # No additional preprocessing needed - computed in feature script
    
    # === New Ecological Features Intermediates ===
    # F1: f_acellular_gap_formation - spread on downticks × top10 net sell ratio
    if "TickSpreadDown" in tick_agg.columns:
        tick_day_spread = tick_agg.groupby(["StockId", "Date"])["TickSpreadDown"].mean().reset_index()
        tick_day_spread.columns = ["StockId", "Date", "raw_mean_spread_downtick"]
        features = features.merge(tick_day_spread, on=["StockId", "Date"], how="left")
        features["raw_mean_spread_downtick"] = features["raw_mean_spread_downtick"].fillna(0)

    # F2 & F3: Refuge/Burst ratios for Top10 brokers
    daily_broker_for_new = daily_broker.copy()
    daily_broker_for_new["top10_Buy"] = np.where(mask_top10, daily_broker_for_new["TotalBuy"], 0.0)
    daily_broker_for_new["top10_RefugeBuy"] = np.where(mask_top10, daily_broker_for_new["EstRefugeBuy"], 0.0)
    daily_broker_for_new["top10_BurstBuy"] = np.where(mask_top10, daily_broker_for_new["EstBurstBuy"], 0.0)
    daily_broker_for_new["top10_LateQuietBuy"] = np.where(mask_top10, daily_broker_for_new["EstLateQuietBuy"], 0.0)

    top10_new = daily_broker_for_new[daily_broker_for_new["top10_Buy"] > 0].groupby(["StockId", "Date"]).agg(
        top10_TotalBuy=("top10_Buy", "sum"),
        top10_RefugeBuy=("top10_RefugeBuy", "sum"),
        top10_BurstBuy=("top10_BurstBuy", "sum"),
        top10_LateQuietBuy=("top10_LateQuietBuy", "sum"),
    ).reset_index()

    features = features.merge(top10_new, on=["StockId", "Date"], how="left")

    features["raw_top10_refuge_ratio"] = features["top10_RefugeBuy"] / (features["top10_TotalBuy"] + 1)

    # F3: f_masting_synchronous_burst - HHI of burst buying
    daily_broker_for_hhi = daily_broker_for_new[daily_broker_for_new["top10_BurstBuy"] > 0].copy()
    daily_broker_for_hhi["burst_share"] = (
        daily_broker_for_hhi["top10_BurstBuy"] / 
        daily_broker_for_hhi.groupby(["StockId", "Date"])["top10_BurstBuy"].transform("sum").replace(0, 1)
    )
    daily_broker_for_hhi["burst_hhi"] = daily_broker_for_hhi["burst_share"] ** 2
    burst_hhi = daily_broker_for_hhi.groupby(["StockId", "Date"])["burst_hhi"].sum().reset_index()
    burst_hhi.columns = ["StockId", "Date", "raw_burst_hhi_top10"]
    features = features.merge(burst_hhi, on=["StockId", "Date"], how="left")
    features["raw_burst_hhi_top10"] = features["raw_burst_hhi_top10"].fillna(0)

    # F5: f_spatial_mixing_refuge - Top5 vs Rest quiet ratios
    daily_broker_for_new["top5_Buy"] = np.where(mask_top5, daily_broker_for_new["TotalBuy"], 0.0)
    daily_broker_for_new["top5_RefugeBuy"] = np.where(mask_top5, daily_broker_for_new["EstLateQuietBuy"], 0.0)

    top5_new = daily_broker_for_new[daily_broker_for_new["top5_Buy"] > 0].groupby(["StockId", "Date"]).agg(
        top5_TotalBuy=("top5_Buy", "sum"),
        top5_LateQuietBuy=("top5_RefugeBuy", "sum"),
    ).reset_index()

    rest_new = daily_broker_for_new[daily_broker_for_new["top5_Buy"] == 0].groupby(["StockId", "Date"]).agg(
        rest_TotalBuy=("TotalBuy", "sum"),
        rest_LateQuietBuy=("EstLateQuietBuy", "sum"),
    ).reset_index()

    features = features.merge(top5_new, on=["StockId", "Date"], how="left")
    features = features.merge(rest_new, on=["StockId", "Date"], how="left")

    features["raw_top5_quiet_ratio"] = features["top5_LateQuietBuy"] / (features["top5_TotalBuy"] + 1)
    features["raw_rest_quiet_ratio"] = features["rest_LateQuietBuy"] / (features["rest_TotalBuy"] + 1)

    for col in ["raw_top10_refuge_ratio", "raw_burst_hhi_top10", "raw_top5_quiet_ratio", "raw_rest_quiet_ratio"]:
        if col in features.columns:
            features[col] = features[col].fillna(0)
    
    # ---- NEW: Merge F26-F40 intermediates ----
    features = features.merge(tick_day_base, on=["StockId", "Date"], how="left")
    features = features.merge(broker_netbuy_agg, on=["StockId", "Date"], how="left")
    features = features.merge(nb_buy_brokers, on=["StockId", "Date"], how="left")
    
    for col in ["raw_tick_ret", "raw_delta_p", "raw_buy_vol", "raw_sell_vol", "raw_total_vol_tick",
                "raw_large_buy", "raw_small_buy", "raw_large_sell", "raw_small_sell",
                "raw_spread_mean", "raw_a_price", "raw_top10_net_buy", "raw_top5_net_buy", "raw_retail_net_buy",
                "raw_total_buy_broker"]:
        if col in features.columns:
            features[col] = features[col].fillna(0)
    
    # ---- NEW: Add rolling slopes for F31 ----
    from feature_engine.utils import rolling_slope
    features = features.sort_values(["StockId", "Date"])
    # NEW code uses data[i-window:i] (preceding window, not including current point)
    # FE rolling_slope uses current point, so we shift by 1 to match NEW code behavior
    features["raw_top10_net_buy_slope_5d"] = features.groupby("StockId")["raw_top10_net_buy"].transform(
        lambda x: rolling_slope(x, window=5, min_periods=3).shift(1).fillna(0)
    )
    features["raw_top10_net_buy_slope_20d"] = features.groupby("StockId")["raw_top10_net_buy"].transform(
        lambda x: rolling_slope(x, window=20, min_periods=10).shift(1).fillna(0)
    )
    features["raw_ret_slope_5d"] = features.groupby("StockId")["raw_tick_ret"].transform(
        lambda x: rolling_slope(x, window=5, min_periods=3).shift(1).fillna(0)
    )
    features["raw_ret_slope_20d"] = features.groupby("StockId")["raw_tick_ret"].transform(
        lambda x: rolling_slope(x, window=20, min_periods=10).shift(1).fillna(0)
    )
    for col in ["raw_top10_net_buy_slope_5d", "raw_top10_net_buy_slope_20d", "raw_ret_slope_5d", "raw_ret_slope_20d"]:
        features[col] = features[col].fillna(0)
    # ---- END NEW ----
    
    return features.reset_index()
