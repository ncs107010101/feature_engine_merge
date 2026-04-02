import pandas as pd
import numpy as np
import time

def _safe_calc(val):
    if pd.isna(val) or np.isinf(val):
        return 0.0
    return float(val)

def _safe_skew(x):
    if len(x) < 3:
        return 0.0
    try:
        s = x.skew()
        return 0.0 if pd.isna(s) else s
    except Exception:
        return 0.0

def _calc_slope(y):
    if len(y) > 2:
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0] / (y.mean() + 1e-8)
    return 0.0

# Copying the current implementation
def preprocess_single_stock_tick(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date_int"] = df["Date"].astype(int)
    # Using TotalQty instead of DealTimeSecond ensures stable deterministic ordering for sub-second precision
    df.sort_values(["StockId", "Date_int", "TotalQty"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["DealValue"] = df["DealCount"] * df["DealPrice"] * 1000
    df["ActiveBuy"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
    df["ActiveSell"] = np.where(df["PrFlag"] == 0, df["DealCount"], 0)
    df["Minute"] = df["DealTimeSecond"] // 60

    # Cumulatives for Bars
    gb_day = df.groupby(["StockId", "Date_int"])
    df["CumVol"] = df.groupby(["StockId", "Date_int"])["DealCount"].cumsum()
    df["CumDol"] = df.groupby(["StockId", "Date_int"])["DealValue"].cumsum()
    df["TickID_Base"] = df.groupby(["StockId", "Date_int"]).cumcount()
    
    df["TickBarId"] = df["TickID_Base"] // 50
    df["VolBarId"] = df["CumVol"] // 500
    df["DolBarId"] = df["CumDol"] // 30_000_000  # Used for Dollar Bar aggregation (line ~336)
    
    # ========== New features: Jump-based calculations ==========
    # Previous tick price for jump calculation (using TotalQty sorted order)
    df["raw_prev_price"] = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(1)
    # Jump percentage: (current price - prev price) / prev price
    df["raw_jump_pct"] = np.where(
        df["raw_prev_price"] > 0,
        (df["DealPrice"] - df["raw_prev_price"]) / df["raw_prev_price"],
        0.0
    )
    # Active buy flag (PrFlag == 1 means buyer initiated = active buy)
    df["raw_is_active_buy"] = (df["PrFlag"] == 1).astype(int)
    # Fat-tail jump: Jump >= 0.3% (30 bps) AND active buy
    df["raw_is_fat_tail_jump"] = ((df["raw_jump_pct"] >= 0.003) & (df["PrFlag"] == 1)).astype(int)
    # Gaussian jump: 0 <= Jump <= 0.1% AND active buy
    df["raw_is_gaussian_jump"] = (
        (df["raw_jump_pct"] >= 0) & (df["raw_jump_pct"] <= 0.001) & (df["PrFlag"] == 1)
    ).astype(int)
    # Levy jump: Jump >= 0.3% AND active buy (same as fat-tail)
    df["raw_is_levy_jump"] = df["raw_is_fat_tail_jump"]
    # ============================================================
    
    # 1. Global Daily Aggregates
    day_agg = gb_day.agg(
        total_vol=("DealCount", "sum"),
        buy_vol=("ActiveBuy", "sum"),
        sell_vol=("ActiveSell", "sum"),
        total_value=("DealValue", "sum"),
        open_pr=("DealPrice", "first"),
        close_pr=("DealPrice", "last"),
        high_pr=("DealPrice", "max"),
        low_pr=("DealPrice", "min"),
        intensity_max=("DealTimeSecond", lambda x: x.value_counts().max() if len(x)>0 else 0),
        tick_count=("DealCount", "size")
    )
    
    # Alpha v17: Compute rolling 20-day thresholds for small/large trade identification
    daily_q30 = df.groupby(["StockId", "Date_int"])["DealCount"].quantile(0.3).reset_index()
    daily_q30.columns = ["StockId", "Date_int", "q30"]
    daily_q80 = df.groupby(["StockId", "Date_int"])["DealCount"].quantile(0.8).reset_index()
    daily_q80.columns = ["StockId", "Date_int", "q80"]
    daily_q = daily_q30.merge(daily_q80, on=["StockId", "Date_int"])
    daily_q = daily_q.sort_values(["StockId", "Date_int"]).reset_index(drop=True)
    daily_q["Small_Thresh"] = daily_q.groupby("StockId")["q30"].transform(
        lambda x: x.rolling(20, min_periods=1).median().shift(1)
    )
    daily_q["Large_Thresh"] = daily_q.groupby("StockId")["q80"].transform(
        lambda x: x.rolling(20, min_periods=1).median().shift(1)
    )
    daily_q["Small_Thresh"] = daily_q["Small_Thresh"].fillna(daily_q["q30"])
    daily_q["Large_Thresh"] = daily_q["Large_Thresh"].fillna(daily_q["q80"])
    daily_q["Small_Thresh"] = daily_q["Small_Thresh"].clip(lower=1)
    daily_q["Large_Thresh"] = daily_q["Large_Thresh"].clip(lower=1)
    thresh_map = daily_q.set_index(["StockId", "Date_int"])[["Small_Thresh", "Large_Thresh"]]
    day_agg = day_agg.join(thresh_map)
    day_agg["Small_Thresh"] = day_agg["Small_Thresh"].fillna(1)
    day_agg["Large_Thresh"] = day_agg["Large_Thresh"].fillna(1)
    thresh_df = thresh_map.reset_index()
    df = df.merge(thresh_df, on=["StockId", "Date_int"], how="left")
    df["Small_Thresh"] = df["Small_Thresh"].fillna(1)
    df["Large_Thresh"] = df["Large_Thresh"].fillna(1)
    
    # Alpha v17: Shared intermediate calculations for tick-level features
    df["_is_buy_vol"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
    df["_is_sell_vol"] = np.where(df["PrFlag"] == 0, df["DealCount"], 0)
    df["_small_buy"] = np.where(
        (df["PrFlag"] == 1) & (df["DealCount"] <= df["Small_Thresh"]),
        df["DealCount"], 0
    )
    df["_small_sell"] = np.where(
        (df["PrFlag"] == 0) & (df["DealCount"] <= df["Small_Thresh"]),
        df["DealCount"], 0
    )
    df["_large_buy"] = np.where(
        (df["PrFlag"] == 1) & (df["DealCount"] >= df["Large_Thresh"]),
        df["DealCount"], 0
    )
    df["_large_sell"] = np.where(
        (df["PrFlag"] == 0) & (df["DealCount"] >= df["Large_Thresh"]),
        df["DealCount"], 0
    )
    if "BuyPr" in df.columns and "SellPr" in df.columns:
        df["_spread"] = (df["SellPr"] - df["BuyPr"]).clip(lower=0)
    else:
        df["_spread"] = 0
    df["_pv"] = df["DealPrice"] * df["DealCount"]
    
    def _trade_skew(x):
        if len(x) >= 5 and np.std(x.values) > 0:
            s = x.skew()
            return 0.0 if pd.isna(s) else float(s)
        return 0.0
    skew_deal = gb_day["DealCount"].apply(_trade_skew)

    # L30 (last 30 min)
    LAST_30MIN_SEC = 13 * 3600
    l30_mask = df["DealTimeSecond"] >= LAST_30MIN_SEC
    l30_agg = df[l30_mask].groupby(["StockId", "Date_int"]).agg(
        l30_buy=("ActiveBuy", "sum"),
        l30_sell=("ActiveSell", "sum"),
        l30_vol=("DealCount", "sum")
    )

    day_agg = day_agg.join(l30_agg)
    day_agg["l30_buy"] = day_agg["l30_buy"].fillna(0)
    day_agg["l30_sell"] = day_agg["l30_sell"].fillna(0)
    day_agg["l30_vol"] = day_agg["l30_vol"].fillna(0)

    # 0218 features
    day_agg["raw_tick_mom"] = day_agg["buy_vol"] / (day_agg["total_vol"] + 1e-9)
    day_agg["raw_vpin"] = abs(day_agg["buy_vol"] - day_agg["sell_vol"]) / (day_agg["total_vol"] + 1e-9)
    day_agg["raw_trade_skew"] = skew_deal
    l30_net = day_agg["l30_buy"] - day_agg["l30_sell"]
    day_agg["raw_close_pressure"] = (l30_net / (day_agg["l30_vol"] + 1e-9)) * (day_agg["l30_vol"] / (day_agg["total_vol"] + 1e-9))

    # 0221 Minute logic
    min_bars = df.groupby(["StockId", "Date_int", "Minute"]).agg(
        first_pr=("DealPrice", "first"),
        last_pr=("DealPrice", "last")
    )
    min_ret = (min_bars["last_pr"] - min_bars["first_pr"]) / min_bars["first_pr"]
    
    surge_count = min_ret[min_ret > 0.005].groupby(["StockId", "Date_int"]).count()
    plunge_count = min_ret[min_ret < -0.005].groupby(["StockId", "Date_int"]).count()
    day_agg["raw_surge_count"] = surge_count.reindex(day_agg.index).fillna(0)
    day_agg["raw_plunge_count"] = plunge_count.reindex(day_agg.index).fillna(0)

    # Morning surge
    morning_mask = df["DealTimeSecond"] <= (32400 + 15 * 60)
    morning_high = df[morning_mask].groupby(["StockId", "Date_int"])["DealPrice"].max()
    morning_high = morning_high.reindex(day_agg.index).fillna(day_agg["open_pr"])
    day_agg["raw_morning_surge"] = np.where(day_agg["open_pr"] > 0, (morning_high - day_agg["open_pr"]) / day_agg["open_pr"], 0.0)

    day_agg["vwap"] = day_agg["total_value"] / (day_agg["total_vol"] * 1000)

    # High zone sell
    df_merged = df.merge(day_agg[["high_pr", "low_pr", "open_pr", "close_pr", "vwap"]], on=["StockId", "Date_int"])
    high_zone = df_merged["DealPrice"] >= df_merged["low_pr"] + 0.9 * (df_merged["high_pr"] - df_merged["low_pr"])
    valid_high_zone = df_merged["high_pr"] > df_merged["low_pr"]
    high_zone_sell = df_merged[valid_high_zone & high_zone & (df_merged["PrFlag"] == 0)].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_high_zone_sell"] = high_zone_sell.reindex(day_agg.index).fillna(0) / day_agg["total_vol"]

    day_agg["raw_intensity_max"] = day_agg["intensity_max"]

    # Large dev
    large_buy = df_merged[(df_merged["PrFlag"] == 1) & (df_merged["DealValue"] >= 1000000)].groupby(["StockId", "Date_int"]).agg(
        val=("DealValue", "sum"),
        c=("DealCount", "sum")
    )
    large_vwap = large_buy["val"] / (large_buy["c"] * 1000)
    day_agg["raw_large_vwap_dev"] = np.where((large_buy["c"].reindex(day_agg.index).fillna(0) > 0) & (day_agg["vwap"] > 0), (large_vwap.reindex(day_agg.index).fillna(day_agg["vwap"]) - day_agg["vwap"]) / day_agg["vwap"], 0.0)

    day_agg["raw_failed_surge"] = np.where((day_agg["raw_surge_count"] > 0) & (day_agg["close_pr"] < day_agg["vwap"]), 1.0, 0.0)

    # Plunge buy
    plunge_mins = min_ret[min_ret < -0.003].reset_index()[["StockId", "Date_int", "Minute"]]
    plunge_mins["StockId"] = plunge_mins["StockId"].astype(str)
    plunge_mins["Date_int"] = plunge_mins["Date_int"].astype(np.int64)
    plunge_mins["Minute"] = plunge_mins["Minute"].astype(np.int64)
    
    df_copy = df[["StockId", "Date_int", "Minute", "DealCount", "ActiveBuy"]].copy()
    df_copy["StockId"] = df_copy["StockId"].astype(str)
    df_copy["Date_int"] = df_copy["Date_int"].astype(np.int64)
    df_copy["Minute"] = df_copy["Minute"].astype(np.int64)

    plunge_ticks = df_copy.merge(plunge_mins, on=["StockId", "Date_int", "Minute"], how="inner")
    plunge_agg = plunge_ticks.groupby(["StockId", "Date_int"]).agg(
        pv=("DealCount", "sum"),
        pb=("ActiveBuy", "sum")
    )
    day_agg["raw_dip_buy"] = (plunge_agg["pb"] / plunge_agg["pv"]).reindex(day_agg.index).fillna(0)

    # ========== New features: Burst and Plunge minute aggregations ==========
    # Minute-level aggregations for f_masting_predator_satiation and f_predator_mate_limitation
    min_agg = df.groupby(["StockId", "Date_int", "Minute"]).agg(
        min_active_buy=("ActiveBuy", "sum"),
        min_active_sell=("ActiveSell", "sum"),
        min_total_vol=("DealCount", "sum")
    ).reset_index()
    
    # f_masting_predator_satiation: max minute active buy volume (burst)
    burst_vol = min_agg.groupby(["StockId", "Date_int"])["min_active_buy"].max()
    day_agg["raw_burst_buy_vol"] = burst_vol.reindex(day_agg.index).fillna(0)
    
    # f_predator_mate_limitation: Plunge minute detection (return < -0.3%)
    min_ret_df = min_ret.reset_index()
    min_ret_df.columns = ["StockId", "Date_int", "Minute", "min_ret"]
    min_agg = min_agg.merge(min_ret_df, on=["StockId", "Date_int", "Minute"], how="left")
    min_agg["is_plunge"] = min_agg["min_ret"] < -0.003
    
    # Calculate HHI for sell volumes in plunge minutes
    plunge_min = min_agg[min_agg["is_plunge"]].copy()
    
    def _calc_hhi(vol_series):
        if len(vol_series) == 0 or vol_series.sum() == 0:
            return 0.0
        shares = vol_series / vol_series.sum()
        return (shares ** 2).sum()
    
    sell_hhi_plunge = plunge_min.groupby(["StockId", "Date_int"])["min_active_sell"].apply(_calc_hhi)
    buy_depth_plunge = plunge_min.groupby(["StockId", "Date_int"])["min_active_sell"].sum()
    
    day_agg["raw_plunge_sell_hhi"] = sell_hhi_plunge.reindex(day_agg.index).fillna(0)
    day_agg["raw_plunge_buy_depth"] = buy_depth_plunge.reindex(day_agg.index).fillna(0)
    # =======================================================================
    
    # ========== New features: Day-level aggregations from tick flags ==========
    # f_fat_tailed_seed_dispersal: fat-tail active buy / total active buy
    fat_tail_vol = df[df["raw_is_fat_tail_jump"] == 1].groupby(["StockId", "Date_int"])["DealCount"].sum()
    total_active_buy = df[df["raw_is_active_buy"] == 1].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_fat_tailed_seed_dispersal"] = (
        fat_tail_vol.reindex(day_agg.index).fillna(0) / 
        (total_active_buy.reindex(day_agg.index).fillna(0) + 1e-8)
    )
    
    # f_mixed_levy_gaussian_dispersion: gaussian_ratio * levy_ratio
    gaussian_vol = df[df["raw_is_gaussian_jump"] == 1].groupby(["StockId", "Date_int"])["DealCount"].sum()
    levy_vol = df[df["raw_is_levy_jump"] == 1].groupby(["StockId", "Date_int"])["DealCount"].sum()
    gaussian_ratio = gaussian_vol.reindex(day_agg.index).fillna(0) / (total_active_buy.reindex(day_agg.index).fillna(0) + 1e-8)
    levy_ratio = levy_vol.reindex(day_agg.index).fillna(0) / (total_active_buy.reindex(day_agg.index).fillna(0) + 1e-8)
    day_agg["raw_mixed_levy_gaussian_dispersion"] = gaussian_ratio * levy_ratio
    # =========================================================================
    
    late_group = df[df["DealTimeSecond"] >= 46800].groupby(["StockId", "Date_int"])["DealPrice"].min()
    late_low = late_group.reindex(day_agg.index).fillna(day_agg["close_pr"])
    day_agg["raw_morning_trap"] = np.where(day_agg["open_pr"] > 0, (morning_high - late_low) / day_agg["open_pr"], 0.0)

    above_vwap = df_merged[df_merged["DealPrice"] > df_merged["vwap"]].groupby(["StockId", "Date_int"]).agg(
        av=("DealCount", "sum"),
        sav=("ActiveSell", "sum")
    )
    day_agg["raw_sell_above_vwap"] = (above_vwap["sav"] / above_vwap["av"]).reindex(day_agg.index).fillna(0)

    large_sell = df_merged[(df_merged["PrFlag"] == 0) & (df_merged["DealValue"] >= 1000000)].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_large_sell"] = (large_sell.reindex(day_agg.index).fillna(0) / day_agg["total_vol"])

    # 0223 OFI & VWAP Trend (Volume Bar level)
    # DealValue_Unscaled = Count * Price (unscaled, for per-bar VWAP calculation)
    # Note: differs from DealValue (= Count * Price * 1000) used for daily-level VWAP.
    # Per-bar VWAP = sum(Count*Price) / sum(Count) = volume-weighted avg price, which is correct.
    df["DealValue_Unscaled"] = df["DealCount"] * df["DealPrice"]
    # Vol Bars
    vb_agg = df.groupby(["StockId", "Date_int", "VolBarId"]).agg(
        vwap_num=("DealValue_Unscaled", "sum"),
        count=("DealCount", "sum"),
        abuy=("ActiveBuy", "sum"),
        asell=("ActiveSell", "sum")
    )
    vbar_agg = vb_agg.copy()
    
    vbar_agg["bar_vwap"] = vbar_agg["vwap_num"] / vbar_agg["count"].replace(0, 1)
    vbar_agg["ofi"] = vbar_agg["abuy"] - vbar_agg["asell"]
    
    ofi_skew = vbar_agg.groupby(["StockId", "Date_int"])["ofi"].apply(_safe_skew)
    vwap_trend = vbar_agg.groupby(["StockId", "Date_int"])["bar_vwap"].apply(_calc_slope)
    day_agg["raw_ofi_skew"] = ofi_skew.reindex(day_agg.index).fillna(0)
    day_agg["raw_vwap_trend"] = vwap_trend.reindex(day_agg.index).fillna(0)

    df["DbarId_0223"] = df["CumDol"] // 50_000_000  # Larger threshold for volatility calc (fewer bars, more stable)
    
    dbar_close_0223 = df.groupby(["StockId", "Date_int", "DbarId_0223"])["DealPrice"].last()
    dbar_ret_0223 = dbar_close_0223.groupby(["StockId", "Date_int"]).pct_change().fillna(0)
    dbar_volatility = dbar_ret_0223.groupby(["StockId", "Date_int"]).std()
        
    day_agg["raw_dbar_vol"] = dbar_volatility.reindex(day_agg.index).fillna(0)

    am_mask = df["DealTimeSecond"] < 10 * 3600
    pm_mask = df["DealTimeSecond"] > 12.5 * 3600
    am_agg = df[am_mask].groupby(["StockId", "Date_int"]).agg(v=("DealCount", "sum"), b=("ActiveBuy", "sum"))
    pm_agg = df[pm_mask].groupby(["StockId", "Date_int"]).agg(v=("DealCount", "sum"), b=("ActiveBuy", "sum"))
    
    am_ratio = (am_agg["b"] / am_agg["v"]).reindex(day_agg.index).fillna(0.5)
    pm_ratio = (pm_agg["b"] / pm_agg["v"]).reindex(day_agg.index).fillna(0.5)
    day_agg["raw_am_pm_div"] = am_ratio - pm_ratio

    late_mask = df["DealTimeSecond"] >= 13 * 3600  # Consistent with LAST_30MIN_SEC
    smart_late_df = df_merged[late_mask & (df_merged["PrFlag"] == 1) & (df_merged["DealPrice"] > df_merged["vwap"])]
    smart_late_buy = smart_late_df.groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_smart_late"] = (smart_late_buy.reindex(day_agg.index).fillna(0) / day_agg["buy_vol"])

    mean_buy_size = df[df["PrFlag"] == 1].groupby(["StockId", "Date_int"])["DealCount"].mean()
    mean_sell_size = df[df["PrFlag"] == 0].groupby(["StockId", "Date_int"])["DealCount"].mean()
    day_agg["raw_size_ratio"] = mean_buy_size.reindex(day_agg.index).fillna(0) / (mean_sell_size.reindex(day_agg.index).fillna(0) + 1e-9)

    # 0225
    day_agg["raw_order_imbalance"] = (day_agg["buy_vol"] - day_agg["sell_vol"]) / (day_agg["total_vol"] + 1e-8)
    early_vol = df[df["DealTimeSecond"] < 9 * 3600 + 30 * 60].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_first_30min_vol_share"] = early_vol.reindex(day_agg.index).fillna(0) / day_agg["total_vol"]

    mmd = np.log((day_agg["high_pr"] - day_agg["close_pr"]) / (day_agg["close_pr"] - day_agg["low_pr"]))
    day_agg["raw_intraday_momentum_decay"] = np.where((day_agg["close_pr"] - day_agg["low_pr"] > 0) & (day_agg["high_pr"] - day_agg["close_pr"] > 0), mmd, 0.0)
    
    prange = day_agg["high_pr"] - day_agg["low_pr"]
    day_agg["raw_close_vs_intraday_high"] = np.where(prange > 0, (day_agg["close_pr"] - day_agg["low_pr"]) / prange, 0.5)

    # === Alpha v2 Custom Aggregations ===
    
    # F3: VWAP Density (Price in VWAP +/- 2%)
    df["_vwap_tmp"] = df.set_index(["StockId", "Date_int"]).index.map(day_agg["vwap"])
    vwap_density_mask = (df["DealPrice"] >= df["_vwap_tmp"] * 0.98) & (df["DealPrice"] <= df["_vwap_tmp"] * 1.02)
    day_agg["raw_vwap_density"] = df[vwap_density_mask].groupby(["StockId", "Date_int"])["DealCount"].sum().reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)

    # F11: Burst Frequency
    mean_size_series = day_agg["total_vol"] / day_agg["tick_count"].replace(0, 1)
    df["_mean_size"] = df.set_index(["StockId", "Date_int"]).index.map(mean_size_series)
    df["IsBurst"] = (df["DealCount"] > 3 * df["_mean_size"]).astype(int)
    bursts_per_min = df.groupby(["StockId", "Date_int", "Minute"])["IsBurst"].sum()
    freq_bursts_per_min = (bursts_per_min >= 2).astype(int)
    num_burst_events = freq_bursts_per_min.groupby(["StockId", "Date_int"]).sum()
    day_agg["raw_burst_freq"] = num_burst_events.reindex(day_agg.index).fillna(0)
    
    # F13: Breakout Velocity (time difference between low and high)
    idx_min = df.groupby(["StockId", "Date_int"])["DealPrice"].idxmin()
    idx_max = df.groupby(["StockId", "Date_int"])["DealPrice"].idxmax()
    idx_min_valid = idx_min.dropna().astype(int)
    idx_max_valid = idx_max.dropna().astype(int)
    time_min = df.loc[idx_min_valid].set_index(["StockId", "Date_int"])["DealTimeSecond"]
    time_max = df.loc[idx_max_valid].set_index(["StockId", "Date_int"])["DealTimeSecond"]
    time_diff = (time_max - time_min).abs() + 1
    day_agg["raw_breakout_time"] = time_diff.reindex(day_agg.index).fillna(1)
    
    # F14: Spread & Volatility
    if "BuyPr" in df.columns and "SellPr" in df.columns:
        valid_spread_mask = (df["BuyPr"] > 0) & (df["SellPr"] > 0)
        spread_vals = np.where(valid_spread_mask, df["SellPr"] - df["BuyPr"], df["DealPrice"] * 0.10)
        df["_spread"] = np.where(spread_vals >= 0, spread_vals, np.nan)
        day_agg["raw_tick_spread"] = df.groupby(["StockId", "Date_int"])["_spread"].mean().reindex(day_agg.index).fillna(0)
    else:
        day_agg["raw_tick_spread"] = (day_agg["high_pr"] - day_agg["low_pr"])
    day_agg["raw_tick_volatility"] = df.groupby(["StockId", "Date_int"])["DealPrice"].std().reindex(day_agg.index).fillna(0)

    # F15: Second Level Momentum Continuity
    df["_diff"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff()
    df["_is_up"] = (df["_diff"] > 0).astype(int)
    df["_is_up_prev"] = df.groupby(["StockId", "Date_int"])["_is_up"].shift(1)
    df["_cont_up"] = ((df["_is_up"] == 1) & (df["_is_up_prev"] == 1)).astype(int)
    day_agg["raw_momentum_continuity"] = (df.groupby(["StockId", "Date_int"])["_cont_up"].sum() / (day_agg["tick_count"] - 2).clip(lower=1)).reindex(day_agg.index).fillna(0.5)

    df.drop(columns=["_vwap_tmp", "_mean_size", "IsBurst", "_spread", "_diff", "_is_up", "_is_up_prev", "_cont_up"], inplace=True, errors="ignore")
    # === End Alpha v2 Custom ===

    # 0224 Bars
    # Tick Bars
    tb_agg = df.groupby(["StockId", "Date_int", "TickBarId"]).agg(
        Close=("DealPrice", "last")
    )
    tb_agg.dropna(subset=["Close"], inplace=True)
    tb_ret = tb_agg.groupby(["StockId", "Date_int"])["Close"].pct_change().fillna(0)
    tb_var = tb_ret.groupby(["StockId", "Date_int"]).var()
    # Variance Ratio: use non-overlapping k-bar blocks (k=10) for unbiased estimate
    tb_ret_reset = tb_ret.reset_index()
    tb_ret_reset["block"] = tb_ret_reset.groupby(["StockId", "Date_int"]).cumcount() // 10
    block_ret = tb_ret_reset.groupby(["StockId", "Date_int", "block"])["Close"].sum()
    tb_var_k = block_ret.groupby(["StockId", "Date_int"]).var() / 10
    
    var_ratio = np.where((tb_agg.groupby(["StockId", "Date_int"]).size() > 10) & (tb_var > 0), tb_var_k.reindex(tb_var.index).fillna(0) / tb_var, 0.0)
    day_agg["raw_tkb_var_ratio"] = pd.Series(var_ratio, index=tb_var.index).fillna(0)

    # Run length - pos returns
    pos_ret = (tb_ret > 0).astype(int)
    # cumcount on consecutive groups
    rl = pos_ret * (pos_ret.groupby(["StockId", "Date_int", (pos_ret != pos_ret.groupby(["StockId", "Date_int"]).shift()).cumsum()]).cumcount() + 1)
    day_agg["raw_tkb_run_length"] = rl.groupby(["StockId", "Date_int"]).max().fillna(0)

    # Reversals
    signs = np.sign(tb_agg.groupby(["StockId", "Date_int"])["Close"].diff().dropna())
    reversals = (signs != signs.groupby(["StockId", "Date_int"]).shift(1)).groupby(["StockId", "Date_int"]).sum()
    n_signs = signs.groupby(["StockId", "Date_int"]).size()
    day_agg["raw_tkb_reversal_freq"] = (reversals / n_signs).fillna(0)

    # Vol Bars
    vb_agg = df.groupby(["StockId", "Date_int", "VolBarId"]).agg(
        Close=("DealPrice", "last"),
        Volume=("DealCount", "sum"),
        ActiveBuyV=("ActiveBuy", "sum"),
        ActiveSellV=("ActiveSell", "sum"),
        DealValueV=("DealValue", "sum"),
        TimeFirst=("DealTimeSecond", "first"),
        TimeLast=("DealTimeSecond", "last"),
        TradeCount=("DealCount", "size"),
        MaxTradeVol=("DealCount", "max"),
    )
    vb_agg.dropna(subset=["Close"], inplace=True)
    vb_agg["Return"] = vb_agg.groupby(["StockId", "Date_int"])["Close"].pct_change().fillna(0)
    vb_agg["Duration"] = vb_agg["TimeLast"] - vb_agg["TimeFirst"]
    vb_agg["OFI_v"] = vb_agg["ActiveBuyV"] - vb_agg["ActiveSellV"]

    dom_mask = (vb_agg["MaxTradeVol"] / (vb_agg["Volume"] + 1e-5)) > 0.3
    day_agg["raw_vb_large_domination"] = dom_mask.groupby(["StockId", "Date_int"]).mean().fillna(0)
    day_agg["raw_vb_sell_intensity_var"] = vb_agg.groupby(["StockId", "Date_int"])["ActiveSellV"].var().fillna(0)
    day_agg["raw_vb_pinning_prob"] = (vb_agg["Return"] == 0).groupby(["StockId", "Date_int"]).mean().fillna(0)
    day_agg["raw_vb_ofi_vol"] = vb_agg.groupby(["StockId", "Date_int"])["OFI_v"].std().fillna(0)

    daily_twap = vb_agg.groupby(["StockId", "Date_int"])["Close"].mean()
    vb_daily_vwap = day_agg["total_value"] / (day_agg["total_vol"] * 1000 + 1e-5)
    
    dev_twap = (vb_daily_vwap / (daily_twap + 1e-5) - 1)
    day_agg["raw_vb_vwap_twap_dev"] = dev_twap.replace([np.inf, -np.inf], 0.0).fillna(0)
    
    final_close = vb_agg.groupby(["StockId", "Date_int"])["Close"].last()
    dev_vwap = (final_close / (vb_daily_vwap + 1e-5) - 1)
    day_agg["raw_vb_close_vwap_dev"] = dev_vwap.replace([np.inf, -np.inf], 0.0).fillna(0)

    # Ext dur ratio
    vb_cnt = vb_agg.groupby(["StockId", "Date_int"]).size()
    top5p = vb_agg.groupby(["StockId", "Date_int"])["Return"].transform(lambda x: x.quantile(0.95))
    bot5p = vb_agg.groupby(["StockId", "Date_int"])["Return"].transform(lambda x: x.quantile(0.05))
    ext_mask = (vb_agg["Return"] >= top5p) | (vb_agg["Return"] <= bot5p)
    ext_dur = vb_agg[ext_mask].groupby(["StockId", "Date_int"])["Duration"].mean()
    norm_dur = vb_agg[~ext_mask].groupby(["StockId", "Date_int"])["Duration"].mean()
    ext_ratio = ext_dur / (norm_dur + 1e-5)
    day_agg["raw_vb_ext_dur_ratio"] = np.where(vb_cnt > 10, ext_ratio.reindex(day_agg.index).fillna(0), 0.0)

    day_agg["raw_vb_trade_dispersion"] = (vb_agg.groupby(["StockId", "Date_int"])["TradeCount"].std() / (vb_agg.groupby(["StockId", "Date_int"])["TradeCount"].mean() + 1e-5)).fillna(0)
    zero_ret_vol = vb_agg[vb_agg["Return"] == 0].groupby(["StockId", "Date_int"])["Volume"].sum()
    day_agg["raw_vb_absorption_ratio"] = (zero_ret_vol.reindex(day_agg.index).fillna(0) / (day_agg["total_vol"] + 1e-5))

    # Dol Bars
    db_agg = df.groupby(["StockId", "Date_int", "DolBarId"]).agg(
        Close=("DealPrice", "last"),
        ActiveBuyD=("ActiveBuy", "sum"),
        ActiveSellD=("ActiveSell", "sum"),
        TimeFirst=("DealTimeSecond", "first"),
        TimeLast=("DealTimeSecond", "last"),
        Volume=("DealCount", "sum")
    )
    db_agg.dropna(subset=["Close"], inplace=True)
    db_agg["Return"] = db_agg.groupby(["StockId", "Date_int"])["Close"].pct_change().fillna(0)
    db_agg["Duration"] = db_agg["TimeLast"] - db_agg["TimeFirst"]
    db_agg["Speed"] = 1.0 / (db_agg["Duration"] + 1e-5)

    def _db_trend(g):
        y = g["Close"].values
        if len(y) > 5:
            x = np.arange(len(y))
            cm = np.corrcoef(x, y)
            if not np.isnan(cm).any():
                return cm[0, 1] ** 2
        return 0.0
    day_agg["raw_db_trend_r2"] = db_agg.groupby(["StockId", "Date_int"]).apply(_db_trend).fillna(0)

    db_buy = db_agg.groupby(["StockId", "Date_int"])["ActiveBuyD"].sum()
    db_sell = db_agg.groupby(["StockId", "Date_int"])["ActiveSellD"].sum()
    day_agg["raw_db_vpin"] = (abs(db_buy - db_sell) / (db_buy + db_sell + 1e-5)).fillna(0)

    neg_rets = db_agg[db_agg["Return"] < 0].groupby(["StockId", "Date_int"])["Return"].apply(lambda x: np.sqrt(np.mean(x**2)))
    day_agg["raw_db_downside_dev"] = neg_rets.reindex(day_agg.index).replace([np.inf, -np.inf], 0.0).fillna(0)

    # RollVol max
    def _roll_max(g):
        w = min(5, len(g))
        return g["Volume"].rolling(w, min_periods=1).sum().max()
    roll_max = db_agg.groupby(["StockId", "Date_int"]).apply(_roll_max)
    avg_dol_vol = db_agg.groupby(["StockId", "Date_int"])["Volume"].mean()
    day_agg["raw_db_peak_vel_ratio"] = (roll_max / (avg_dol_vol + 1e-5)).fillna(0)

    # High vel vol
    fast = db_agg.groupby(["StockId", "Date_int"])["Speed"].transform(lambda x: x.quantile(0.90))
    fast_bars = db_agg[db_agg["Speed"] >= fast].groupby(["StockId", "Date_int"])["Return"].apply(lambda x: x.abs().sum())
    db_cnt = db_agg.groupby(["StockId", "Date_int"]).size()
    db_cnt_aligned = db_cnt.reindex(day_agg.index).fillna(0)
    day_agg["raw_db_high_vel_vol"] = np.where(db_cnt_aligned > 10, fast_bars.reindex(day_agg.index).replace([np.inf, -np.inf], 0.0).fillna(0), 0.0)

    # === Alpha v3 19 New Tick Aggregations ===
    
    # 1. f_price_bimodality
    # Weighted BC is complex to fully vectorise without explode. We approximate by taking price variance and skew directly on DealPrice.
    # Note: original repeats prices by volume. We can use weighted stats.
    def _w_bimodality(g):
        p = g["DealPrice"].values
        v = g["DealCount"].values
        if len(p) < 10 or v.sum() == 0: return np.nan
        w = v / v.sum()
        mu = np.sum(w * p)
        diff = p - mu
        var = np.sum(w * diff**2)
        if var < 1e-12: return np.nan
        std = np.sqrt(var)
        skew = np.sum(w * (diff/std)**3)
        kurt = np.sum(w * (diff/std)**4)
        return (skew**2 + 1) / (kurt + 1e-10)
    
    day_agg["raw_price_bimodality"] = df.groupby(["StockId", "Date_int"]).apply(_w_bimodality).reindex(day_agg.index).values

    # 2. f_collapse_direction
    def _collapse(g):
        t = g["DealTimeSecond"].values
        p = g["DealPrice"].values
        v = g["DealCount"].values
        if len(p) < 10: return np.nan
        t_split = t.min() + (t.max() - t.min()) * 0.8
        em, lm = t < t_split, t >= t_split
        if em.sum() < 5 or lm.sum() < 3: return np.nan
        ev, lv = v[em].sum(), v[lm].sum()
        if ev == 0 or lv == 0: return np.nan
        mid_p = np.sum(p[em]*v[em]) / ev
        late_v = np.sum(p[lm]*v[lm]) / lv
        return (late_v - mid_p) / (mid_p + 1e-10)
    day_agg["raw_collapse_direction"] = df.groupby(["StockId", "Date_int"]).apply(_collapse).reindex(day_agg.index).values

    # 3. f_belief_shock
    def _belief_shock(g):
        t, p, v, f = g["DealTimeSecond"].values, g["DealPrice"].values, g["DealCount"].values, g["PrFlag"].values
        if len(v) < 20: return np.nan
        thr = np.percentile(v, 95)
        if thr <= 0: return np.nan
        l_idx = np.where(v >= thr)[0]
        if len(l_idx) == 0: return np.nan
        
        shocks, valids = 0, 0
        for idx in l_idx:
            flag = f[idx]
            if flag == 2: continue
            fut = np.where((t > t[idx]) & (t <= t[idx] + 60))[0]
            if len(fut) == 0: continue
            valids += 1
            move = p[fut[-1]] - p[idx]
            if flag == 1 and move < 0: shocks += 1
            elif flag == 0 and move > 0: shocks += 1
        return shocks / valids if valids > 0 else np.nan
    day_agg["raw_belief_shock"] = df.groupby(["StockId", "Date_int"]).apply(_belief_shock).reindex(day_agg.index).values

    # 4. f_integer_level_density
    df["_is_round"] = False
    p_val = df["DealPrice"].values
    m1 = p_val < 10; df.loc[m1, "_is_round"] = (np.round(p_val[m1]*100).astype(np.int64) % 100) == 0
    m2 = (p_val >= 10) & (p_val < 50); df.loc[m2, "_is_round"] = (np.round(p_val[m2]*20).astype(np.int64) % 10) == 0
    m3 = (p_val >= 50) & (p_val < 100); df.loc[m3, "_is_round"] = (np.round(p_val[m3]*10).astype(np.int64) % 10) == 0
    m4 = (p_val >= 100) & (p_val < 500); df.loc[m4, "_is_round"] = (np.round(p_val[m4]*2).astype(np.int64) % 10) == 0
    m5 = (p_val >= 500) & (p_val < 1000); df.loc[m5, "_is_round"] = (np.round(p_val[m5]).astype(np.int64) % 10) == 0
    m6 = (p_val >= 1000); df.loc[m6, "_is_round"] = (np.round(p_val[m6]).astype(np.int64) % 50) == 0
    round_vol = np.where(df["_is_round"], df["DealCount"], 0)
    round_vol = np.where(df["_is_round"], df["DealCount"], 0)
    df["_round_vol"] = round_vol
    day_agg["raw_integer_level_density"] = (df.groupby(["StockId", "Date_int"])["_round_vol"].sum() / (day_agg["total_vol"] + 1e-10)).reindex(day_agg.index).values
    df.drop(columns=["_is_round", "_round_vol"], inplace=True)

    # 5. f_trade_quantization
    valid_q = (df["DealCount"] > 0) & (df["DealCount"] < 500)
    df["_is_mult5"] = np.where(valid_q, (df["DealCount"] % 5 == 0).astype(float), np.nan)
    day_agg["raw_trade_quantization"] = df.groupby(["StockId", "Date_int"])["_is_mult5"].mean().reindex(day_agg.index).fillna(0).values
    df.drop(columns=["_is_mult5"], inplace=True)

    # 6. f_contextual_momentum_conflict
    def _conflict(g):
        p, v, t = g["DealPrice"].values, g["DealCount"].values, g["DealTimeSecond"].values
        n = len(p)
        if n < 30: return np.nan
        
        def _slope(bt, n_bars=9):
            if n < n_bars*2: return 0
            if bt == 'time':
                tmin, tmax = t.min(), t.max()
                if tmin == tmax: return 0
                edges = np.linspace(tmin, tmax+1, n_bars+1)
                bidx = np.clip(np.digitize(t, edges)-1, 0, n_bars-1)
            elif bt == 'volume':
                cv = np.cumsum(v)
                if cv[-1] == 0: return 0
                edges = np.linspace(0, cv[-1], n_bars+1)[1:]
                bidx = np.searchsorted(cv, edges, side='right')
                ba = np.zeros(n, dtype=int)
                prev = 0
                for i in range(n_bars):
                    end = min(bidx[i] if i < len(bidx) else n, n)
                    ba[prev:end] = i
                    prev = end
                bidx = ba
            else: # tick
                bidx = np.clip(np.arange(n) // max(1, n//n_bars), 0, n_bars-1)
                
            vwaps = []
            for b in range(n_bars):
                m = bidx == b
                vb, pb = v[m], p[m]
                vs = vb.sum()
                vwaps.append(np.sum(pb*vb)/vs if vs > 0 else np.nan)
            vwaps = np.array(vwaps)
            tail = vwaps[n_bars*2//3:]
            tail = tail[~np.isnan(tail)]
            if len(tail) < 2: return 0
            return np.sign(tail[-1] - tail[0])
            
        signs = [_slope('time'), _slope('volume'), _slope('tick')]
        return (3 - abs(sum(signs))) / 2.0
    day_agg["raw_contextual_momentum_conflict"] = df.groupby(["StockId", "Date_int"]).apply(_conflict).reindex(day_agg.index).values

    # 7. f_tick_entropy_rate
    def _entropy(g):
        flags = g["PrFlag"].values
        flags = flags[(flags == 0) | (flags == 1)]
        n = len(flags)
        if n < 20: return np.nan
        
        # Build 3-grams
        from collections import Counter
        trigrams = [(flags[i], flags[i+1], flags[i+2]) for i in range(n - 2)]
        if len(trigrams) == 0: return np.nan
        
        counts = Counter(trigrams)
        total = sum(counts.values())
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy / 3.0
    day_agg["raw_tick_entropy_rate"] = df.groupby(["StockId", "Date_int"]).apply(_entropy).reindex(day_agg.index).values

    # 8. f_volume_acceleration
    def _accel(g):
        t, v = g["DealTimeSecond"].values, g["DealCount"].values
        if len(t) < 20: return np.nan
        edges = list(range(32400, 48601, 1800))
        bidx = np.clip(np.digitize(t, edges)-1, 0, len(edges)-2)
        ivols = np.zeros(len(edges)-1)
        for b in range(len(ivols)): ivols[b] = v[bidx==b].sum()
        if (ivols > 0).sum() < 3: return np.nan
        acc = np.diff(np.diff(ivols))
        if len(acc) < 2: return np.nan
        avg_v = ivols.mean()
        if avg_v < 1: return np.nan
        return acc[-2:].mean() / avg_v
    day_agg["raw_volume_acceleration"] = df.groupby(["StockId", "Date_int"]).apply(_accel).reindex(day_agg.index).values

    # 9. f_price_return_asymmetry
    def _asymm(g):
        prices = g["DealPrice"].values
        times = g["DealTimeSecond"].values
        if len(prices) < 20: return np.nan
        rets = np.diff(prices) / (prices[:-1] + 1e-10)
        t_gaps = np.clip(np.diff(times), 0, 300).astype(float)
        up = t_gaps[rets > 1e-8]
        dn = t_gaps[rets < -1e-8]
        if len(up) < 5 or len(dn) < 5: return np.nan
        avg_dn = dn.mean()
        if avg_dn < 0.01: return np.nan
        return np.log(up.mean() / (avg_dn + 1e-10))
    day_agg["raw_price_return_asymmetry"] = df.groupby(["StockId", "Date_int"]).apply(_asymm).reindex(day_agg.index).values

    # 10. f_vwap_gravity_pull
    def _gravity(g):
        p, v = g["DealPrice"].values, g["DealCount"].values
        if len(p) < 30: return np.nan
        rv = np.cumsum(p*v) / (np.cumsum(v) + 1e-10)
        dev = (p - rv) / (rv + 1e-10)
        if len(dev) < 10: return np.nan
        dev_c = dev - dev.mean()
        var = np.sum(dev_c**2)
        if var < 1e-15: return np.nan
        return np.sum(dev_c[:-1] * dev_c[1:]) / var
    day_agg["raw_vwap_gravity_pull"] = df.groupby(["StockId", "Date_int"]).apply(_gravity).reindex(day_agg.index).values

    # 11. f_order_flow_momentum (Tail OFI slope)
    def _ofi_slope(g):
        t, c, f = g["DealTimeSecond"].values, g["DealCount"].values, g["PrFlag"].values
        if len(t) < 20: return np.nan
        sign = np.where(f==1, 1.0, np.where(f==0, -1.0, 0.0))
        ofi_cum = np.cumsum(c * sign)
        mask = t >= 46800
        t_tail, ofi_tail = t[mask], ofi_cum[mask]
        if len(t_tail) < 10: return np.nan
        x = (t_tail - t_tail[0]).astype(float)
        y = ofi_tail.astype(float)
        xm, ym = x.mean(), y.mean()
        ssxx = np.sum((x-xm)**2)
        if ssxx < 1e-10: return np.nan
        slope = np.sum((x-xm)*(y-ym)) / ssxx
        vs = c.sum()
        if vs < 1: return np.nan
        return slope / vs * 1800
    day_agg["raw_order_flow_momentum"] = df.groupby(["StockId", "Date_int"]).apply(_ofi_slope).reindex(day_agg.index).values

    # 12. f_intraday_reversal_depth
    def _rev_depth(g):
        p, t = g["DealPrice"].values, g["DealTimeSecond"].values
        if len(p) < 10: return np.nan
        hp, lp = p.max(), p.min()
        if hp <= lp or lp < 1e-6: return np.nan
        th, tl = t[np.argmax(p)], t[np.argmin(p)]
        swing = (hp - lp) / lp
        return swing if tl < th else -swing
    day_agg["raw_intraday_reversal_depth"] = df.groupby(["StockId", "Date_int"]).apply(_rev_depth).reindex(day_agg.index).values

    # 13. f_tick_run_imbalance
    def _run_imb(g):
        flags = g['PrFlag'].values
        valid = (flags == 0) | (flags == 1)
        flags = flags[valid]
        if len(flags) < 20: return np.nan
        def _mrun(arr, tgt):
            if len(arr) == 0: return 0
            m = np.r_[False, arr == tgt, False]
            idx = np.flatnonzero(m[:-1] != m[1:])
            lens = idx[1::2] - idx[::2]
            return lens.max() if len(lens) > 0 else 0
        mb = _mrun(flags, 1)
        ms = _mrun(flags, 0)
        den = mb + ms
        return (mb - ms) / den if den > 0 else np.nan
    day_agg["raw_tick_run_imbalance"] = df.groupby(["StockId", "Date_int"]).apply(_run_imb).reindex(day_agg.index).values

    # 14. f_spread_compression
    if "BuyPr" in df.columns and "SellPr" in df.columns:
        bpr, spr, rpr = df["BuyPr"].values, df["SellPr"].values, df["RefPr"].values
        sprd = spr - bpr
        valid_s = (bpr > 0) & (spr > 0) & (sprd >= 0)
        df["_sv"] = np.where(valid_s, sprd, np.nan)
        df["_rv"] = np.where(valid_s, rpr, np.nan)
        med_s = df.groupby(["StockId", "Date_int"])["_sv"].median()
        med_r = df.groupby(["StockId", "Date_int"])["_rv"].median()
        day_agg["raw_spread_compression"] = (med_s / med_r).reindex(day_agg.index).values
        df.drop(columns=["_sv", "_rv"], inplace=True)
    else:
        day_agg["raw_spread_compression"] = 0.0

    # 15. f_large_trade_timing
    def _timing(g):
        c, t = g["DealCount"].values, g["DealTimeSecond"].values
        if len(c) < 20: return np.nan
        thr = np.percentile(c, 90)
        lm = c >= max(thr, 1)
        lt = t[lm]
        if len(lt) < 3: return np.nan
        edges = np.linspace(32400, 48600, 11)
        bidx = np.clip(np.digitize(lt, edges)-1, 0, 9)
        bc = np.zeros(10)
        for b in bidx: bc[b] += 1
        ts = bc.sum()
        if ts == 0: return np.nan
        return np.sum((bc/ts)**2)
    day_agg["raw_large_trade_timing"] = df.groupby(["StockId", "Date_int"]).apply(_timing).reindex(day_agg.index).values

    # 16. f_early_late_volume_ratio
    ev = df[(df["DealTimeSecond"] >= 32400) & (df["DealTimeSecond"] < 37800)].groupby(["StockId", "Date_int"])["DealCount"].sum()
    lv = df[(df["DealTimeSecond"] >= 43200) & (df["DealTimeSecond"] <= 48600)].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_early_late_volume_ratio"] = np.log((ev.reindex(day_agg.index).fillna(0) + 1) / (lv.reindex(day_agg.index).fillna(0) + 1)).values

    # 17. f_volume_price_divergence
    def _divergence(g):
        t, p, v = g["DealTimeSecond"].values, g["DealPrice"].values, g["DealCount"].values
        if len(p) < 30: return np.nan
        edges = np.linspace(32400, 48600, 11)
        bidx = np.clip(np.digitize(t, edges)-1, 0, 9)
        vwaps, vsums = np.zeros(10), np.zeros(10)
        for b in range(10):
            m = bidx == b
            vb, pb = v[m], p[m]
            vs = vb.sum()
            vsums[b] = vs
            vwaps[b] = np.sum(pb*vb)/vs if vs > 0 else np.nan
        valid = (~np.isnan(vwaps)) & (vsums > 0)
        if valid.sum() < 5: return np.nan
        
        x, y = vwaps[valid], vsums[valid]
        rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
        rxm, rym = rx.mean(), ry.mean()
        sx, sy = np.sqrt(np.sum((rx-rxm)**2)), np.sqrt(np.sum((ry-rym)**2))
        if sx < 1e-10 or sy < 1e-10: return np.nan
        return np.sum((rx-rxm)*(ry-rym)) / (sx*sy)
    day_agg["raw_volume_price_divergence"] = df.groupby(["StockId", "Date_int"]).apply(_divergence).reindex(day_agg.index).values

    # 18. f_tail_session_ofi_zscore
    df_tail = df[df["DealTimeSecond"] >= 46800].copy()
    if not df_tail.empty:
        df_tail["_sign"] = np.where(df_tail["PrFlag"]==1, 1.0, np.where(df_tail["PrFlag"]==0, -1.0, 0.0))
        df_tail["_ofi"] = df_tail["DealCount"] * df_tail["_sign"]
        t_ofi = df_tail.groupby(["StockId", "Date_int"])["_ofi"].sum()
        t_vol = df_tail.groupby(["StockId", "Date_int"])["DealCount"].sum()
        t_cnt = df_tail.groupby(["StockId", "Date_int"]).size()
        raw_tail_ofi = np.where(t_cnt >= 5, t_ofi / (t_vol + 1e-10), np.nan)
        day_agg["raw_tail_session_ofi"] = pd.Series(raw_tail_ofi, index=t_ofi.index).reindex(day_agg.index).values
    else:
        day_agg["raw_tail_session_ofi"] = 0.0

    # 19. f_intraday_volatility_clustering
    def _vol_cluster(g):
        p = g["DealPrice"].values
        if len(p) < 30: return np.nan
        ar = np.abs(np.diff(p) / (p[:-1] + 1e-10))
        if len(ar) < 20: return np.nan
        x, y = ar[:-1], ar[1:]
        xm, ym = x.mean(), y.mean()
        vx, vy = np.sum((x-xm)**2), np.sum((y-ym)**2)
        denom = np.sqrt(vx*vy)
        if denom < 1e-15: return np.nan
        return np.sum((x-xm)*(y-ym)) / denom
    day_agg["raw_intraday_volatility_clustering"] = df.groupby(["StockId", "Date_int"]).apply(_vol_cluster).reindex(day_agg.index).values

    # === Alpha v3 ===
    
    # === Alpha v4 Physics Features ===
    physics_day_agg = pd.DataFrame(index=day_agg.index)
    
    # 1. f_eth_nonadiabatic_entropy_rate
    df["_cum_vwap"] = np.where(df["CumVol"] > 0, df["CumDol"] / (df["CumVol"] * 1000.0 + 1e-8), df["DealPrice"])
    df["_prev_vwap"] = df.groupby(["StockId", "Date_int"])["_cum_vwap"].shift(1)
    df["_vwap_shift"] = (df["_cum_vwap"] - df["_prev_vwap"]).abs() / (df["_prev_vwap"] + 1e-8)
    df["_is_eff_vwap"] = df["_vwap_shift"] > 0.001
    eff_vol = df.loc[df["_is_eff_vwap"], "DealCount"].groupby([df.loc[df["_is_eff_vwap"], "StockId"], df.loc[df["_is_eff_vwap"], "Date_int"]]).sum()
    day_agg["raw_eth_nonadiabatic_entropy_rate"] = (eff_vol.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values
    df.drop(columns=["_cum_vwap", "_prev_vwap", "_vwap_shift", "_is_eff_vwap"], inplace=True)

    # 2. Minute bar aggregations for physics features
    phys_min = df.groupby(["StockId", "Date_int", "Minute"])[["DealPrice", "DealCount"]].agg(
        bar_last=("DealPrice", "last"),
        bar_mean=("DealPrice", "mean"),
        bar_std=("DealPrice", "std"),
        bar_vol=("DealCount", "sum"),
        bar_high=("DealPrice", "max"),
        bar_low=("DealPrice", "min")
    ).reset_index()
    phys_min["bar_std"] = phys_min["bar_std"].fillna(0)
    
    # Pre-calculated shifts
    phys_min["prev_last"] = phys_min.groupby(["StockId", "Date_int"])["bar_last"].shift(1)
    phys_min["ret_abs"] = np.where(phys_min["prev_last"] > 0, (phys_min["bar_last"] - phys_min["prev_last"]).abs() / phys_min["prev_last"], 0)
    phys_min["ret"] = np.where(phys_min["prev_last"] > 0, (phys_min["bar_last"] - phys_min["prev_last"]) / phys_min["prev_last"], 0)
    phys_min["diff"] = phys_min["bar_last"] - phys_min["prev_last"]
    
    # f_eth_adiabatic_maintenance
    day_hl = phys_min.groupby(["StockId", "Date_int"]).agg(day_high=("bar_mean", "max"), day_low=("bar_mean", "min"))
    phys_min = phys_min.merge(day_hl, on=["StockId", "Date_int"])
    phys_min["is_hz"] = phys_min["bar_mean"] >= (phys_min["day_low"] + 0.9 * (phys_min["day_high"] - phys_min["day_low"]))
    phys_min["is_sw"] = phys_min["bar_std"] <= ((phys_min["day_high"] - phys_min["day_low"]) * 0.05)
    phys_min["is_adia"] = phys_min["is_hz"] & phys_min["is_sw"]
    adia_vol = phys_min[phys_min["is_adia"]].groupby(["StockId", "Date_int"])["bar_vol"].sum()
    day_agg["raw_eth_adiabatic_maintenance"] = (adia_vol.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values

    # f_eth_limit_cycle_dissipation
    def _ac1(x):
        if len(x) > 2 and x.std() > 1e-8: return float(x.autocorr(lag=1))
        return 0.0
    ac_vals = phys_min.groupby(["StockId", "Date_int"])["ret"].apply(_ac1)
    day_agg["raw_eth_limit_cycle_dissipation"] = (np.where(ac_vals < 0, -ac_vals, 0) * day_agg["total_vol"]).fillna(0)
    
    # f_eth_active_translational_dissipation
    df["_prev_p"] = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(1)
    conds = [df["_prev_p"] < 10, df["_prev_p"] < 50, df["_prev_p"] < 100, df["_prev_p"] < 500, df["_prev_p"] < 1000]
    chcs = [0.01, 0.05, 0.1, 0.5, 1.0]
    df["_tsize"] = np.select(conds, chcs, default=5.0)
    df["_pd"] = (df["DealPrice"] - df["_prev_p"]).abs()
    df["_is_trans"] = df["_pd"] >= (2.0 * df["_tsize"] - 1e-5)
    trans_v = df.loc[df["_is_trans"], "DealCount"].groupby([df.loc[df["_is_trans"], "StockId"], df.loc[df["_is_trans"], "Date_int"]]).sum()
    day_agg["raw_eth_active_translational_dissipation"] = (trans_v.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values
    df.drop(columns=["_prev_p", "_tsize", "_pd", "_is_trans"], inplace=True)

    # f_flu_instanton_escape_likelihood
    df["_prev_p"] = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(1)
    df["_is_up"] = (df["DealPrice"] >= df["_prev_p"]) & (df["PrFlag"] == 1)
    df["_streak"] = (~df["_is_up"]).groupby([df["StockId"], df["Date_int"]]).cumsum()
    up_streaks = df[df["_is_up"]].groupby(["StockId", "Date_int", "_streak"])["DealCount"].sum()
    max_streak = up_streaks.groupby(["StockId", "Date_int"]).max()
    day_agg["raw_flu_instanton_escape_likelihood"] = (max_streak.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values
    df.drop(columns=["_prev_p", "_is_up", "_streak"], inplace=True)

    # f_flu_fluctuation_asymmetry
    df["_next_p"] = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(-1)
    df["_b_up"] = (df["PrFlag"] == 1) & (df["_next_p"] > df["DealPrice"])
    df["_s_dn"] = (df["PrFlag"] == 0) & (df["_next_p"] < df["DealPrice"])
    n_b_up = df.groupby(["StockId", "Date_int"])["_b_up"].sum()
    n_s_dn = df.groupby(["StockId", "Date_int"])["_s_dn"].sum()
    day_agg["raw_flu_fluctuation_asymmetry"] = (n_b_up.reindex(day_agg.index).fillna(0) / (n_s_dn.reindex(day_agg.index).fillna(0) + 1)).values
    df.drop(columns=["_next_p", "_b_up", "_s_dn"], inplace=True)

    # f_flu_tur_bound_saturation
    df["_chunk"] = ((df["DealTimeSecond"] - 32400) // 540).clip(lower=0, upper=29)
    cks = df.groupby(["StockId", "Date_int", "_chunk"])["DealPrice"].last().reset_index()
    cks["_pp"] = cks.groupby(["StockId", "Date_int"])["DealPrice"].shift(1)
    cks["_r"] = np.where(cks["_pp"] > 0, (cks["DealPrice"] - cks["_pp"]) / cks["_pp"], 0)
    c_agg = cks.groupby(["StockId", "Date_int"]).agg(sr=("_r", "sum"), vr=("_r", "var"))
    day_agg["raw_flu_tur_bound_saturation"] = ((c_agg["sr"].fillna(0)**2) / (c_agg["vr"].fillna(0) + 1e-8)).reindex(day_agg.index).fillna(0).values
    df.drop(columns=["_chunk"], inplace=True)
    
    # f_flu_time_asymmetry
    up_ms = phys_min[phys_min["diff"] > 0].groupby(["StockId", "Date_int"]).size()
    dn_ms = phys_min[phys_min["diff"] < 0].groupby(["StockId", "Date_int"]).size()
    day_agg["raw_flu_time_asymmetry"] = (up_ms.reindex(day_agg.index).fillna(0) / (dn_ms.reindex(day_agg.index).fillna(0) + 1)).values
    
    # f_flu_detailed_balance_breaking
    if "RefPr" in df.columns:
        df["_in_z"] = (df["DealPrice"] >= df["RefPr"] * 0.99) & (df["DealPrice"] <= df["RefPr"] * 1.01)
        b_z = df[df["_in_z"] & (df["PrFlag"] == 1)].groupby(["StockId", "Date_int"])["DealCount"].sum()
        s_z = df[df["_in_z"] & (df["PrFlag"] == 0)].groupby(["StockId", "Date_int"])["DealCount"].sum()
        day_agg["raw_flu_detailed_balance_breaking"] = (b_z.reindex(day_agg.index).fillna(0) / (s_z.reindex(day_agg.index).fillna(0) + 1)).values
        df.drop(columns=["_in_z"], inplace=True)
    else:
        day_agg["raw_flu_detailed_balance_breaking"] = 0.0

    # f_att_attractor_transition_rate
    df["_m5"] = df["DealTimeSecond"] // 300
    b5 = df.groupby(["StockId", "Date_int", "_m5"])["DealPrice"].last().reset_index()
    b5["_q33"] = b5.groupby(["StockId", "Date_int"])["DealPrice"].transform(lambda x: x.quantile(0.33))
    b5["_q67"] = b5.groupby(["StockId", "Date_int"])["DealPrice"].transform(lambda x: x.quantile(0.67))
    cds_b5 = [b5["DealPrice"] <= b5["_q33"], b5["DealPrice"] > b5["_q67"]]
    b5["_st"] = np.select(cds_b5, [0, 2], default=1)
    b5["_pst"] = b5.groupby(["StockId", "Date_int"])["_st"].shift(1)
    b5["_is_tr"] = (b5["_st"] != b5["_pst"]) & (b5["_pst"].notna())
    n_tr = b5[b5["_is_tr"]].groupby(["StockId", "Date_int"]).size()
    n_tot = b5.groupby(["StockId", "Date_int"]).size()
    day_agg["raw_att_attractor_transition_rate"] = (n_tr.reindex(day_agg.index).fillna(0) / n_tot.reindex(day_agg.index).replace(0, 1)).values
    df.drop(columns=["_m5"], inplace=True)

    # f_att_min_entropy_deviation
    phys_min["e_prod"] = phys_min["bar_std"] * phys_min["bar_vol"]
    idx_min = phys_min.groupby(["StockId", "Date_int"])["e_prod"].idxmin().dropna().astype(int)
    qp = phys_min.loc[idx_min.values].set_index(["StockId", "Date_int"])["bar_mean"]
    day_agg["raw_att_min_entropy_deviation"] = np.where(qp.reindex(day_agg.index).fillna(0) > 0, (day_agg["close_pr"] - qp.reindex(day_agg.index).fillna(0)).abs() / qp.reindex(day_agg.index).replace(0, 1), 0.0)

    # f_att_phase_space_contraction
    if "SellPr" in df.columns and "BuyPr" in df.columns:
        v_q = df[(df["SellPr"] > 0) & (df["BuyPr"] > 0)].copy()
        v_q["Spread"] = (v_q["SellPr"] - v_q["BuyPr"]).clip(lower=0)
        v_q["px_m"] = v_q["DealTimeSecond"] // 60
        mb_sq = v_q.groupby(["StockId", "Date_int", "px_m"]).agg(ms=("Spread", "mean"), cnt=("DealTimeSecond", "size")).reset_index()
        mb_sq["pv"] = mb_sq["ms"] * mb_sq["cnt"]
        mpv = mb_sq[mb_sq["px_m"] < 600].groupby(["StockId", "Date_int"])["pv"].mean()
        lpv = mb_sq[mb_sq["px_m"] >= 750].groupby(["StockId", "Date_int"])["pv"].mean()
        day_agg["raw_att_phase_space_contraction"] = (lpv.reindex(day_agg.index).fillna(0) / (mpv.reindex(day_agg.index).fillna(0) + 1e-8)).values
    else:
        day_agg["raw_att_phase_space_contraction"] = 0.0

    df["_vwap_tmp"] = df.set_index(["StockId", "Date_int"]).index.map(day_agg["vwap"]).fillna(0).values

    # f_att_quasipotential_well
    df["_dev_pct"] = np.where(df["_vwap_tmp"] > 0, (df["DealPrice"] - df["_vwap_tmp"]).abs() / df["_vwap_tmp"], 0)
    max_dev = df.groupby(["StockId", "Date_int"])["_dev_pct"].max()
    day_agg["raw_att_quasipotential_well"] = (day_agg["total_vol"] / (max_dev.reindex(day_agg.index).fillna(0) * 10000.0 + 1)).values
    df.drop(columns=["_dev_pct"], inplace=True)

    # f_att_saddle_point_barrier
    v_prof = df.groupby(["StockId", "Date_int", "DealPrice"])["DealCount"].sum().reset_index()
    def _saddle(grp):
        tv = grp["DealCount"].sum()
        if tv == 0 or len(grp) < 3: return 0.0
        gs = grp.sort_values("DealCount", ascending=False)
        t1 = gs.iloc[0]
        t2 = None
        for i in range(1, len(gs)):
            c = gs.iloc[i]
            if abs(c["DealPrice"] - t1["DealPrice"]) / (t1["DealPrice"] + 1e-8) > 0.01:
                t2 = c; break
        if t2 is None: return 0.0
        minp, maxp = min(t1["DealPrice"], t2["DealPrice"]), max(t1["DealPrice"], t2["DealPrice"])
        bw = grp[(grp["DealPrice"] > minp) & (grp["DealPrice"] < maxp)]
        return bw["DealCount"].min() / tv if not bw.empty else 0.0
    s_bar = v_prof.groupby(["StockId", "Date_int"]).apply(_saddle)
    day_agg["raw_att_saddle_point_barrier"] = s_bar.reindex(day_agg.index).fillna(0).values

    # f_dpt_arrhenius_escape_rate
    if "CeilPr" in df.columns and "FloorPr" in df.columns:
        cls_df = df.groupby(["StockId", "Date_int"]).last()[["DealPrice", "CeilPr", "FloorPr"]]
        cls_df["dc"] = np.where(cls_df["DealPrice"] > 0, (cls_df["CeilPr"] - cls_df["DealPrice"]) / cls_df["DealPrice"], 1)
        cls_df["df"] = np.where(cls_df["DealPrice"] > 0, (cls_df["DealPrice"] - cls_df["FloorPr"]) / cls_df["DealPrice"], 1)
        md = np.minimum(cls_df["dc"], cls_df["df"]).clip(lower=0)
        dtk = md * 100
        arr = np.where(md < 0.015, np.exp(-dtk / (day_agg["total_vol"] / 1000.0 + 1e-5)), 0)
        day_agg["raw_dpt_arrhenius_escape_rate"] = pd.Series(arr, index=cls_df.index).reindex(day_agg.index).fillna(0).values
    else:
        day_agg["raw_dpt_arrhenius_escape_rate"] = 0.0

    # f_dpt_nonreciprocal_interaction
    phys_min["_pv"] = phys_min.groupby(["StockId", "Date_int"])["bar_vol"].shift(1)
    phys_min["_pra"] = phys_min.groupby(["StockId", "Date_int"])["ret_abs"].shift(1)
    def _nonrec(g):
        g = g.dropna(subset=["bar_vol", "ret_abs", "_pv", "_pra"])
        if len(g) < 10: return 0.0
        c1 = g["_pv"].corr(g["ret_abs"]) if (g["_pv"].std() > 0 and g["ret_abs"].std() > 0) else 0.0
        c2 = g["_pra"].corr(g["bar_vol"]) if (g["_pra"].std() > 0 and g["bar_vol"].std() > 0) else 0.0
        return float(c1 - c2)
    day_agg["raw_dpt_nonreciprocal_interaction"] = phys_min.groupby(["StockId", "Date_int"]).apply(_nonrec).reindex(day_agg.index).fillna(0).values

    # f_dpt_phase_transition_singularity
    min_vol = phys_min.groupby(["StockId", "Date_int"])["bar_vol"]
    max_sl = min_vol.max()
    mean_sl = min_vol.mean()
    day_agg["raw_dpt_phase_transition_singularity"] = (max_sl.reindex(day_agg.index).fillna(0) / (mean_sl.reindex(day_agg.index).fillna(0) + 1e-8)).values

    # f_dpt_subexponential_responsiveness
    df["_pp"] = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(1)
    df["_pj"] = (df["DealPrice"] - df["_pp"]).abs()
    st = df[df["DealCount"] <= 1].groupby(["StockId", "Date_int"]).agg(mj=('_pj', 'mean'), mp=('DealPrice', 'mean'))
    day_agg["raw_dpt_subexponential_responsiveness"] = np.where(st["mp"].reindex(day_agg.index).fillna(0) > 0, st["mj"].reindex(day_agg.index).fillna(0) / st["mp"].reindex(day_agg.index).replace(0, 1), 0.0)
    df.drop(columns=["_pp", "_pj"], inplace=True)

    # f_dpt_surprisal_derivative
    dmt = df.groupby(["StockId", "Date_int"])["DealCount"].max()
    dmt_roll = dmt.groupby(level="StockId").apply(lambda x: x.rolling(20, min_periods=1).max().reset_index(level=0, drop=True).shift(1)).fillna(0)
    df["_rmax20"] = df.set_index(["StockId", "Date_int"]).index.map(dmt_roll).fillna(0).values
    df["_is_surp"] = (df["DealCount"] > 0.8 * df["_rmax20"]) & (df["_rmax20"] > 0)
    ts = df[(df["DealTimeSecond"] >= 45000) & df["_is_surp"]].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_dpt_surprisal_derivative"] = ts.reindex(day_agg.index).fillna(0).values
    df.drop(columns=["_rmax20", "_is_surp"], inplace=True)

    # f_mac_kramers_moyal_error
    sk_mt = df.groupby(["StockId", "Date_int"])["DealCount"].skew()
    def _kurt(x):
        return float(x.kurtosis()) if len(x) > 3 and x.std() > 0 else 0.0
    kt_mt = df.groupby(["StockId", "Date_int"])["DealCount"].apply(_kurt)
    day_agg["raw_mac_kramers_moyal_error"] = (sk_mt.reindex(day_agg.index).fillna(0) * kt_mt.reindex(day_agg.index).fillna(0)).values

    # f_mac_probability_velocity_circulation
    df["_dv_tmp"] = df["DealPrice"] * df["DealCount"]
    _vwap_precise = df.groupby(["StockId", "Date_int"])["_dv_tmp"].sum() / df.groupby(["StockId", "Date_int"])["DealCount"].sum()
    df["_vwap_precise"] = df.set_index(["StockId", "Date_int"]).index.map(_vwap_precise).fillna(0).values

    lb = df[(df["DealPrice"] < df["_vwap_precise"]) & (df["PrFlag"] == 1)].groupby(["StockId", "Date_int"])["DealCount"].sum()
    hs = df[(df["DealPrice"] > df["_vwap_precise"]) & (df["PrFlag"] == 0)].groupby(["StockId", "Date_int"])["DealCount"].sum()
    av = df[df["PrFlag"].isin([0, 1])].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_mac_probability_velocity_circulation"] = np.where(av.reindex(day_agg.index).fillna(0) > 0, (lb.reindex(day_agg.index).fillna(0) + hs.reindex(day_agg.index).fillna(0)) / av.reindex(day_agg.index).replace(0, 1), 0.0)
    df.drop(columns=["_dv_tmp", "_vwap_precise"], inplace=True)

    # f_mac_temperature_mobility
    phys_min["mob"] = (phys_min["bar_high"] - phys_min["bar_low"]) / (phys_min["bar_vol"] + 1)
    day_tm = phys_min.groupby(["StockId", "Date_int"])["bar_std"].mean()
    tail_mob = phys_min[phys_min["Minute"] >= 750].groupby(["StockId", "Date_int"])["mob"].mean() # Minute = DealTimeSecond // 60, 12:30=750
    day_agg["raw_mac_temperature_mobility"] = (tail_mob.reindex(day_agg.index).fillna(0) / (day_tm.reindex(day_agg.index).fillna(0) + 1e-8)).values
    
    # Clean up df temporary variables
    df.drop(columns=["_vwap_tmp"], inplace=True, errors="ignore")
    # === End Alpha v4 Physics ===

    # === Alpha v5 QIT Physics Features (20) ===

    # 1. f_qit_wigner_negativity_depth
    phys_min["wigner_is_anomaly"] = (phys_min["ret"].abs() < 0.0001) & (phys_min["bar_vol"] > 2 * phys_min.groupby(["StockId", "Date_int"])["bar_vol"].transform("mean"))
    wigner_vol = phys_min[phys_min["wigner_is_anomaly"]].groupby(["StockId", "Date_int"])["bar_vol"].sum()
    day_agg["raw_qit_wigner_negativity_depth"] = (wigner_vol.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values

    # 2. f_qit_teleportation_fidelity
    phys_min_m = phys_min[(phys_min["Minute"] >= 540) & (phys_min["Minute"] < 570)].copy()
    phys_min_l = phys_min[(phys_min["Minute"] >= 780) & (phys_min["Minute"] < 810)].copy()
    phys_min_m["offset"] = phys_min_m["Minute"] - 540
    phys_min_l["offset"] = phys_min_l["Minute"] - 780
    tele_m_sq = (phys_min_m["ret"]**2).groupby([phys_min_m["StockId"], phys_min_m["Date_int"]]).sum()
    tele_l_sq = (phys_min_l["ret"]**2).groupby([phys_min_l["StockId"], phys_min_l["Date_int"]]).sum()
    tele_merge = pd.merge(phys_min_m[["StockId", "Date_int", "offset", "ret"]], phys_min_l[["StockId", "Date_int", "offset", "ret"]], on=["StockId", "Date_int", "offset"], suffixes=("_m", "_l"))
    tele_dot = (tele_merge["ret_m"] * tele_merge["ret_l"]).groupby([tele_merge["StockId"], tele_merge["Date_int"]]).sum()
    m_count = phys_min_m.groupby(["StockId", "Date_int"]).size()
    l_count = phys_min_l.groupby(["StockId", "Date_int"]).size()
    
    norm_m = np.sqrt(tele_m_sq)
    norm_l = np.sqrt(tele_l_sq)
    val_tele = tele_dot / (norm_m * norm_l).replace(0, 1)
    valid_tele = (m_count >= 10) & (l_count >= 10) & (norm_m > 0) & (norm_l > 0)
    day_agg["raw_qit_teleportation_fidelity"] = np.where(valid_tele.reindex(day_agg.index).fillna(False), val_tele.reindex(day_agg.index).fillna(0), 0.0)

    # 3. f_qit_subadditivity_entropy_gap
    from sklearn.metrics import mutual_info_score
    df["_pdiff"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff().fillna(0)
    df["_dir"] = np.sign(df["_pdiff"]).astype(int)
    df["_sclass"] = np.where(df["DealCount"] <= 2, 0, np.where(df["DealCount"] <= 10, 1, 2))
    def _mi_score(g):
        if len(g) < 5: return 0.0
        return mutual_info_score(g["_dir"], g["_sclass"])
    day_agg["raw_qit_subadditivity_entropy_gap"] = df.groupby(["StockId", "Date_int"]).apply(_mi_score).reindex(day_agg.index).fillna(0.0).values
    df.drop(columns=["_pdiff", "_dir", "_sclass"], inplace=True)

    # 4. f_qit_quantum_zeno_freezing
    sec_group = df.groupby(["StockId", "Date_int", "DealTimeSecond"]).agg(
        TickCount=("DealPrice", "count"),
        PriceStd=("DealPrice", "std")
    ).reset_index()
    sec_group["PriceStd"] = sec_group["PriceStd"].fillna(0)
    sec_group["is_frozen"] = (sec_group["TickCount"] >= 3) & (sec_group["PriceStd"] == 0)
    zeno_froz = sec_group.groupby(["StockId", "Date_int"])["is_frozen"].sum()
    zeno_tot = sec_group.groupby(["StockId", "Date_int"]).size()
    day_agg["raw_qit_quantum_zeno_freezing"] = (zeno_froz.reindex(day_agg.index).fillna(0) / zeno_tot.reindex(day_agg.index).fillna(1)).values

    # 5. f_qit_otoc_scrambling_rate
    def _otoc(g):
        if len(g) < 5: return np.nan
        r = g["bar_last"].pct_change()
        return r.corr(r.shift(1))
    day_agg["raw_qit_otoc_scrambling_rate"] = phys_min.groupby(["StockId", "Date_int"]).apply(_otoc).reindex(day_agg.index).fillna(0.0).values

    # 6. f_qit_majorization_deficit
    def _gini(array):
        if len(array) == 0: return 0.0
        array = np.sort(np.array(array, dtype=np.float64))
        s = np.sum(array)
        if s == 0: return 0.0
        n = array.shape[0]
        return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * array) / (n * s))
    active_b = df[df["PrFlag"] == 1]
    mb = active_b[(active_b["DealTimeSecond"] >= 32400) & (active_b["DealTimeSecond"] <= 36000)]
    lb = active_b[(active_b["DealTimeSecond"] >= 45000) & (active_b["DealTimeSecond"] <= 48600)]
    g_m = mb.groupby(["StockId", "Date_int"])["DealCount"].apply(_gini)
    g_l = lb.groupby(["StockId", "Date_int"])["DealCount"].apply(_gini)
    day_agg["raw_qit_majorization_deficit"] = (g_l.reindex(day_agg.index).fillna(0) - g_m.reindex(day_agg.index).fillna(0)).values

    # 7. f_qit_magic_state_injection_ratio
    df["_mean_vol"] = df.groupby(["StockId", "Date_int"])["DealCount"].transform("mean")
    df["_prev_p"] = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(1).fillna(df["DealPrice"])
    df["_pj"] = (df["DealPrice"] - df["_prev_p"]).abs() / df["_prev_p"].replace(0, 1)
    df["_is_magic"] = (df["_pj"] > 0.005) & (df["DealCount"] > 2 * df["_mean_vol"])
    magic_vol = df[df["_is_magic"]].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_qit_magic_state_injection_ratio"] = (magic_vol.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values
    df.drop(columns=["_mean_vol", "_prev_p", "_pj", "_is_magic"], inplace=True)

    # 8. f_qit_lyapunov_ballistic_duration
    df["_prev_p"] = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(1).fillna(0)
    valid_lya = (df["PrFlag"] == 1) & (df["DealPrice"] >= df["_prev_p"])
    df["_lya_block"] = (~valid_lya).groupby([df["StockId"], df["Date_int"]]).cumsum()
    lya_vol = df[valid_lya].groupby(["StockId", "Date_int", "_lya_block"])["DealCount"].sum()
    day_agg["raw_qit_lyapunov_ballistic_duration"] = lya_vol.groupby(["StockId", "Date_int"]).max().reindex(day_agg.index).fillna(0.0).values
    df.drop(columns=["_prev_p", "_lya_block"], inplace=True)

    # 9. f_qit_incompatible_observables_product
    day_range = day_agg["high_pr"] - day_agg["low_pr"]
    pos = np.where(day_range > 0, (day_agg["close_pr"] - day_agg["low_pr"]) / day_range, 0.5)
    late_b = df[(df["PrFlag"] == 1) & (df["DealTimeSecond"] >= 47700)].groupby(["StockId", "Date_int"])["DealCount"].sum()
    mom = np.where(day_agg["buy_vol"] > 0, late_b.reindex(day_agg.index).fillna(0) / day_agg["buy_vol"], 0.0)
    day_agg["raw_qit_incompatible_observables_product"] = pos * mom

    # 10. f_qit_holevo_capacity_utilization
    phys_min["bar_open"] = df.groupby(["StockId", "Date_int", "Minute"])["DealPrice"].first().values
    phys_min["bar_abs"] = (phys_min["bar_last"] - phys_min["bar_open"]).abs()
    tot_path = phys_min.groupby(["StockId", "Date_int"])["bar_abs"].sum()
    real_disp = (day_agg["close_pr"] - day_agg["open_pr"]).abs()
    day_agg["raw_qit_holevo_capacity_utilization"] = np.where(tot_path.reindex(day_agg.index).fillna(0) > 0, real_disp / tot_path.reindex(day_agg.index).fillna(1), 0.0)

    # 11. f_qit_hawking_evaporation_rate
    hz = day_agg["low_pr"] + (day_agg["high_pr"] - day_agg["low_pr"]) * 0.9
    df["_hz"] = df.set_index(["StockId", "Date_int"]).index.map(hz)
    df["_is_hz"] = df["DealPrice"] >= df["_hz"]
    df["_is_hz_sell"] = df["_is_hz"] & (df["PrFlag"] == 0)
    hz_vol = df[df["_is_hz"]].groupby(["StockId", "Date_int"])["DealCount"].sum()
    hz_sell = df[df["_is_hz_sell"]].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_qit_hawking_evaporation_rate"] = np.where(hz_vol.reindex(day_agg.index).fillna(0) > 0, hz_sell.reindex(day_agg.index).fillna(0) / hz_vol.reindex(day_agg.index).fillna(1), 0.0)
    df.drop(columns=["_hz", "_is_hz", "_is_hz_sell"], inplace=True)

    # 12. f_qit_gravito_magnetic_induction
    phys_min["ab_vol"] = df[df["PrFlag"] == 1].groupby(["StockId", "Date_int", "Minute"])["DealCount"].sum().reindex(phys_min.set_index(["StockId", "Date_int", "Minute"]).index).fillna(0).values
    phys_min["active_secs"] = df.groupby(["StockId", "Date_int", "Minute"])["DealTimeSecond"].nunique().reindex(phys_min.set_index(["StockId", "Date_int", "Minute"]).index).fillna(0).values
    phys_min["_vwap"] = phys_min.set_index(["StockId", "Date_int"]).index.map(day_agg["vwap"])
    valid_gmi = (phys_min["_vwap"] > 0) & (phys_min["active_secs"] > 0)
    phys_min["gmi"] = np.where(valid_gmi, (phys_min["ab_vol"] / phys_min["active_secs"].replace(0,1)) * ((phys_min["bar_last"] - phys_min["_vwap"]) / phys_min["_vwap"].replace(0,1)), 0.0)
    day_agg["raw_qit_gravito_magnetic_induction"] = phys_min.groupby(["StockId", "Date_int"])["gmi"].max().reindex(day_agg.index).fillna(0.0).values

    # 13. f_qit_gie_entanglement_witness
    phys_min["CumVWAP_last"] = df.groupby(["StockId", "Date_int", "Minute"])["CumDol"].last().values / np.clip(df.groupby(["StockId", "Date_int", "Minute"])["CumVol"].last().values * 1000, 1e-8, None)
    phys_min["VWAP_Dev"] = (phys_min["bar_last"] - phys_min["CumVWAP_last"]) / np.clip(phys_min["CumVWAP_last"], 1e-8, None)
    def _corr_vol_dev(g):
        if len(g) < 5: return 0.0
        if g["bar_vol"].std() == 0 or g["VWAP_Dev"].std() == 0: return 0.0
        return g["bar_vol"].corr(g["VWAP_Dev"])
    day_agg["raw_qit_gie_entanglement_witness"] = phys_min.groupby(["StockId", "Date_int"]).apply(_corr_vol_dev).reindex(day_agg.index).fillna(0.0).values

    # 14. f_qit_flow_embezzlement_proxy
    df["_prev_p"] = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(1).fillna(0)
    df["_is_up"] = df["DealPrice"] > df["_prev_p"]
    df["_tg"] = df["_is_up"] & (df["PrFlag"] == 1) & (df["DealCount"] <= 2)
    tot_up = df.groupby(["StockId", "Date_int"])["_is_up"].sum()
    tg_up = df.groupby(["StockId", "Date_int"])["_tg"].sum()
    day_agg["raw_qit_flow_embezzlement_proxy"] = np.where(tot_up.reindex(day_agg.index).fillna(0) > 0, tg_up.reindex(day_agg.index).fillna(0) / tot_up.reindex(day_agg.index).fillna(1), 0.0)
    df.drop(columns=["_prev_p", "_is_up", "_tg"], inplace=True)

    # 15. f_qit_dfs_isolation_score
    dfs_vol = df[df["PrFlag"] == 2].groupby(["StockId", "Date_int"])["DealCount"].sum()
    day_agg["raw_qit_dfs_isolation_score"] = (dfs_vol.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values

    # 16. f_qit_dense_coding_capacity
    if "BuyPr" in df.columns and "SellPr" in df.columns:
        df["_s_sell"] = df.groupby(["StockId", "Date_int"])["SellPr"].shift(1).fillna(np.inf)
        df["_s_buy"] = df.groupby(["StockId", "Date_int"])["BuyPr"].shift(1).fillna(0)
        valid_b = (df["_s_sell"] > 0) & (df["_s_buy"] > 0)
        df["_is_dc"] = valid_b & ((df["DealPrice"] > df["_s_sell"]) | (df["DealPrice"] < df["_s_buy"]))
        dc_vol = df[df["_is_dc"]].groupby(["StockId", "Date_int"])["DealCount"].sum()
        day_agg["raw_qit_dense_coding_capacity"] = (dc_vol.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values
        df.drop(columns=["_s_sell", "_s_buy", "_is_dc"], inplace=True)
    else:
        day_agg["raw_qit_dense_coding_capacity"] = 0.0

    # 17. f_qit_decoherence_spin_echo
    phys_min["min_ret"] = (phys_min["bar_last"] - phys_min["bar_open"]) / phys_min["bar_open"].replace(0, 1)
    phys_min["is_plunge"] = phys_min["min_ret"] < -0.005
    c1 = phys_min.groupby(["StockId", "Date_int"])["bar_last"].shift(-1)
    c2 = phys_min.groupby(["StockId", "Date_int"])["bar_last"].shift(-2)
    c3 = phys_min.groupby(["StockId", "Date_int"])["bar_last"].shift(-3)
    cmax = pd.concat([c1, c2, c3], axis=1).max(axis=1)
    phys_min["is_echo"] = phys_min["is_plunge"] & (cmax > phys_min["bar_open"])
    echo_c = phys_min.groupby(["StockId", "Date_int"])["is_echo"].sum()
    day_agg["raw_qit_decoherence_spin_echo"] = echo_c.reindex(day_agg.index).fillna(0.0).values

    # 18. f_qit_bremsstrahlung_tick_radiation
    phys_min["is_plunge2"] = phys_min["min_ret"] < -0.003
    phys_min["MinTime"] = df.groupby(["StockId", "Date_int", "Minute"])["DealTimeSecond"].min().values
    phys_min["start_t"] = phys_min["MinTime"] + 60
    phys_min["end_t"] = phys_min["start_t"] + 30
    
    df_local = df
    def count_rads(g):
        p_date = g[g["is_plunge2"]]
        if p_date.empty: return 0
        stock_id, date_int = g.name
        df_sd = df_local[(df_local["StockId"] == stock_id) & (df_local["Date_int"] == date_int) & (df_local["DealCount"] == 1)]
        if df_sd.empty: return 0
        ans = 0
        times = df_sd["DealTimeSecond"].values
        for st, et in zip(p_date["start_t"].values, p_date["end_t"].values):
            ans += ((times > st) & (times <= et)).sum()
        return ans
    day_agg["raw_qit_bremsstrahlung_tick_radiation"] = phys_min.groupby(["StockId", "Date_int"]).apply(count_rads).reindex(day_agg.index).fillna(0.0).values

    # 19. f_qit_arnold_web_resonance
    pvol = df.groupby(["StockId", "Date_int", "DealPrice"])["DealCount"].sum()
    max_pvol = pvol.groupby(["StockId", "Date_int"]).max()
    day_agg["raw_qit_arnold_web_resonance"] = (max_pvol.reindex(day_agg.index).fillna(0) / day_agg["total_vol"].replace(0, 1)).values

    # 20. f_qit_virasoro_worldsheet_area
    df["_min_y"] = df.set_index(["StockId", "Date_int"]).index.map(day_agg["low_pr"])
    df["_y_abv"] = df["DealPrice"] - df["_min_y"]
    df["_sh_y"] = df.groupby(["StockId", "Date_int"])["_y_abv"].shift(1).fillna(df["_y_abv"])
    df["_avg_y"] = (df["_y_abv"] + df["_sh_y"]) / 2.0
    def _area_apply(x): return (x["_avg_y"] * x["DealCount"]).sum()
    area_contrib = df.groupby(["StockId", "Date_int"]).apply(_area_apply)
    ideal_area = day_agg["total_vol"] * (day_agg["high_pr"] - day_agg["low_pr"])
    day_agg["raw_qit_virasoro_worldsheet_area"] = np.where(ideal_area > 0, area_contrib.reindex(day_agg.index).fillna(0) / ideal_area.replace(0, 1), 0.0)
    df.drop(columns=["_min_y", "_y_abv", "_sh_y", "_avg_y"], inplace=True)

    # === End Alpha v5 QIT Physics Features ===

    # === Alpha v6 PHY Physics Features (16) ===
    # df is already sorted by [StockId, Date_int, TotalQty] — no re-sort needed
    # RefPr.replace(0,1) avoids div-by-zero without changing values (RefPr > 0 always in practice)

    # --- Shared single-pass intermediates (compute once, reuse below) ---
    df["_phy_ref"] = df["RefPr"].replace(0, 1)
    df["_phy_dP"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff().fillna(0)
    df["_phy_TR"] = df["_phy_dP"] / df["_phy_ref"]
    df["_phy_avg_vol"] = (
        df.groupby(["StockId", "Date_int"])["DealCount"]
        .transform("mean")
        .replace(0, 1)
    )
    df["_phy_VR"] = df["DealCount"] / df["_phy_avg_vol"]
    df["_phy_FS"] = np.where(df["PrFlag"] == 1, 1, np.where(df["PrFlag"] == 0, -1, 0))

    # 1. f_phy_casimir_vacuum_pressure_gradient
    #    sum( TickRet / (TickVolRatio + 1e-4) )  -- price movement on tiny volume
    df["_cas"] = df["_phy_TR"] / (df["_phy_VR"] + 1e-4)
    day_agg["raw_phy_casimir_vacuum_pressure_gradient"] = (
        df.groupby(["StockId", "Date_int"])["_cas"].sum()
        .reindex(day_agg.index).fillna(0).values
    )

    # 2. f_phy_vacancy_induced_gapless_mode
    #    sum( TickVolRatio * sign(TickRet) * (TickRet*1000)^2 )  -- jump amplification
    df["_vac"] = df["_phy_VR"] * np.sign(df["_phy_TR"]) * ((df["_phy_TR"] * 1000) ** 2)
    day_agg["raw_phy_vacancy_induced_gapless_mode"] = (
        df.groupby(["StockId", "Date_int"])["_vac"].sum()
        .reindex(day_agg.index).fillna(0).values
    )

    # 3. f_phy_wignersmith_phase_derivative_q
    #    sum( TickRet * TickVolRatio * AutoCorrMask )  -- keeps only same-direction consecutive ticks
    _prev_tr_sign = np.sign(
        df.groupby(["StockId", "Date_int"])["_phy_TR"].shift(1).fillna(0)
    )
    df["_wigner_mask"] = np.where(
        np.sign(df["_phy_TR"]) * _prev_tr_sign > 0, 1, 0
    )
    df["_wigner"] = df["_phy_TR"] * df["_phy_VR"] * df["_wigner_mask"]
    day_agg["raw_phy_wignersmith_phase_derivative_q"] = (
        df.groupby(["StockId", "Date_int"])["_wigner"].sum()
        .reindex(day_agg.index).fillna(0).values
    )

    # 4. f_phy_dcdw_collinear_acceleration
    #    0.5 * (TickRet*TickFlow + |TickRet*TickFlow|) * sign(TickRet)  -- only same-sign contributions
    df["_TickFlow"] = df["_phy_VR"] * df["_phy_FS"]
    _ret_flow = df["_phy_TR"] * df["_TickFlow"]
    df["_collin"] = 0.5 * (_ret_flow + np.abs(_ret_flow)) * np.sign(df["_phy_TR"])
    day_agg["raw_phy_dcdw_collinear_acceleration"] = (
        df.groupby(["StockId", "Date_int"])["_collin"].sum()
        .reindex(day_agg.index).fillna(0).values
    )

    # 5. f_phy_dislocation_glide_climb_bias
    #    log(R_down / R_up)  R_up=BuyVol/upRet, R_down=SellVol/downRet
    df["_TR_up"] = np.where(df["_phy_TR"] > 0, df["_phy_TR"], 0)
    df["_TR_dn"] = np.where(df["_phy_TR"] < 0, np.abs(df["_phy_TR"]), 0)
    df["_BuyVR"] = np.where(df["PrFlag"] == 1, df["_phy_VR"], 0)
    df["_SellVR"] = np.where(df["PrFlag"] == 0, df["_phy_VR"], 0)
    _dis_agg = df.groupby(["StockId", "Date_int"]).agg(
        _sbvr=("_BuyVR", "sum"),
        _srup=("_TR_up", "sum"),
        _ssvr=("_SellVR", "sum"),
        _srdn=("_TR_dn", "sum"),
    )
    _R_up = np.clip(_dis_agg["_sbvr"] / (_dis_agg["_srup"] + 1e-6), 1e-6, None)
    _R_dn = np.clip(_dis_agg["_ssvr"] / (_dis_agg["_srdn"] + 1e-6), 1e-6, None)
    day_agg["raw_phy_dislocation_glide_climb_bias"] = (
        np.log(_R_dn / _R_up)
        .reindex(day_agg.index)
        .replace([np.inf, -np.inf], 0)
        .fillna(0).values
    )

    # 6. f_phy_neutrino_energy_degradation_regeneration
    #    Inertia = sum(TickVolRatio) / (sum(|TickRet|) + 1e-4); Output = Inertia * IntraRet
    df["_TR_abs"] = np.abs(df["_phy_TR"])
    _neu_agg = df.groupby(["StockId", "Date_int"]).agg(
        _svr=("_phy_VR", "sum"),
        _sra=("_TR_abs", "sum"),
        _open=("DealPrice", "first"),
        _close=("DealPrice", "last"),
    )
    _Inertia = _neu_agg["_svr"] / (_neu_agg["_sra"] + 1e-4)
    _IntraRet_neu = (
        (_neu_agg["_close"] - _neu_agg["_open"])
        / (_neu_agg["_open"].replace(0, 1) + 1e-6)
    )
    day_agg["raw_phy_neutrino_energy_degradation_regeneration"] = (
        (_Inertia * _IntraRet_neu)
        .reindex(day_agg.index)
        .replace([np.inf, -np.inf], 0)
        .fillna(0).values
    )

    # 7. f_phy_tidal_trapping_breakout
    #    Trap = sum( TickVolRatio / (|DealPrice - VWAP|/RefPr + 0.001) ); Output = Trap * IntraRet
    df["_Amt"] = df["DealPrice"] * df["DealCount"]
    _tid_agg = df.groupby(["StockId", "Date_int"]).agg(
        _ta=("_Amt", "sum"),
        _tv=("DealCount", "sum"),
        _op=("DealPrice", "first"),
        _cl=("DealPrice", "last"),
    )
    _VWAP_tid = _tid_agg["_ta"] / _tid_agg["_tv"].replace(0, 1)
    df["_VWAP_t"] = (
        df.set_index(["StockId", "Date_int"]).index.map(_VWAP_tid).values
    )
    df["_DevRet"] = np.abs(df["DealPrice"] - df["_VWAP_t"]) / df["_phy_ref"]
    df["_TrapComp"] = df["_phy_VR"] / (df["_DevRet"] + 0.001)
    _Trap = df.groupby(["StockId", "Date_int"])["_TrapComp"].sum()
    _TidIntraRet = (
        (_tid_agg["_cl"] - _tid_agg["_op"])
        / (_tid_agg["_op"].replace(0, 1) + 1e-6)
    )
    day_agg["raw_phy_tidal_trapping_breakout"] = (
        (_Trap * _TidIntraRet)
        .reindex(day_agg.index)
        .replace([np.inf, -np.inf], 0)
        .fillna(0).values
    )

    # 8. f_phy_fractional_charge_excitation
    #    sum( TickVolRatio * (ConsecTicks/day_tick_count) * PrFlagSign )  -- SCALE INVARIANT fix
    df["_PCh"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff().fillna(0)
    df["_BlkId"] = (
        (df["_PCh"] != 0).astype(int)
        .groupby([df["StockId"], df["Date_int"]])
        .cumsum()
    )
    df["_ConsecTicks"] = (
        df.groupby(["StockId", "Date_int", "_BlkId"]).cumcount() + 1
    )
    # Normalize by total tick count of the day (price-scale invariance fix)
    _day_tick_count = (
        df.groupby(["StockId", "Date_int"])["DealCount"]
        .transform("size")
        .replace(0, 1)
    )
    df["_NormConsecTicks"] = df["_ConsecTicks"] / _day_tick_count
    df["_frac"] = df["_phy_VR"] * df["_NormConsecTicks"] * df["_phy_FS"]
    day_agg["raw_phy_fractional_charge_excitation"] = (
        df.groupby(["StockId", "Date_int"])["_frac"].sum()
        .reindex(day_agg.index).fillna(0).values
    )

    # 9. f_phy_phase_matching_resonance
    #    |AutoCorr_1min_NetBuy| * Daily_NetBuyRatio
    #    HHMMSS is "HH:MM:SS" string → take first 5 chars "HH:MM" as minute key
    if "HHMMSS" in df.columns:
        df["_HHMM"] = df["HHMMSS"].astype(str).str[:5]
        df["_NetBuyCnt"] = df["DealCount"] * df["_phy_FS"]
        _min_df = (
            df.groupby(["StockId", "Date_int", "_HHMM"])["_NetBuyCnt"]
            .sum()
            .reset_index()
        )

        def _ac1(s):
            return s.autocorr(lag=1) if len(s) >= 2 else 0.0

        _ac = (
            _min_df.groupby(["StockId", "Date_int"])["_NetBuyCnt"]
            .apply(_ac1)
            .fillna(0)
        )
        _daily_nb_agg = df.groupby(["StockId", "Date_int"]).agg(
            _tnb=("_NetBuyCnt", "sum"),
            _tv_ph=("DealCount", "sum"),
        )
        _daily_nbr = _daily_nb_agg["_tnb"] / (_daily_nb_agg["_tv_ph"] + 1e-6)
        day_agg["raw_phy_phase_matching_resonance"] = (
            (np.abs(_ac) * _daily_nbr)
            .reindex(day_agg.index)
            .fillna(0).values
        )
    else:
        day_agg["raw_phy_phase_matching_resonance"] = 0.0

    # 10. f_phy_optical_depth_breakthrough
    #    sum( (CumVol_at_Price.shift(1) / day_avg_vol) * TickRet )
    df["_CumVol_P"] = (
        df.groupby(["StockId", "Date_int", "DealPrice"])["DealCount"].cumsum()
    )
    df["_PrevCumVol_P"] = (
        df.groupby(["StockId", "Date_int", "DealPrice"])["_CumVol_P"]
        .shift(1)
        .fillna(0)
    )
    df["_DepthRatio"] = df["_PrevCumVol_P"] / df["_phy_avg_vol"]
    df["_optical"] = df["_DepthRatio"] * df["_phy_TR"]
    day_agg["raw_phy_optical_depth_breakthrough"] = (
        df.groupby(["StockId", "Date_int"])["_optical"].sum()
        .reindex(day_agg.index).fillna(0).values
    )

    # 11. f_phy_landau_peierls_quench
    #    RevFreq (rolling 50 within day) * TickVolRatio * PrFlagSign
    _prev_tr_sign2 = np.sign(
        df.groupby(["StockId", "Date_int"])["_phy_TR"].shift(1).fillna(0)
    )
    df["_IsRev"] = ((np.sign(df["_phy_TR"]) * _prev_tr_sign2) < 0).astype(int)
    df["_RevFreq"] = (
        df.groupby(["StockId", "Date_int"])["_IsRev"]
        .rolling(window=50, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    df["_quench"] = df["_RevFreq"] * df["_phy_VR"] * df["_phy_FS"]
    day_agg["raw_phy_landau_peierls_quench"] = (
        df.groupby(["StockId", "Date_int"])["_quench"].sum()
        .reindex(day_agg.index).fillna(0).values
    )

    # 12-16: Features requiring BuyPr / SellPr
    if "BuyPr" in df.columns and "SellPr" in df.columns:
        df["_Spread"] = df["SellPr"] - df["BuyPr"]
        df["_MidP"] = (df["BuyPr"] + df["SellPr"]) / 2.0
        df["_dSpread"] = (
            df.groupby(["StockId", "Date_int"])["_Spread"].diff().fillna(0)
        )
        df["_dMidP"] = (
            df.groupby(["StockId", "Date_int"])["_MidP"].diff().fillna(0)
        )
        df["_dBuyPr"] = (
            df.groupby(["StockId", "Date_int"])["BuyPr"].diff().fillna(0)
        )
        df["_dSellPr"] = (
            df.groupby(["StockId", "Date_int"])["SellPr"].diff().fillna(0)
        )

        # 12. f_phy_ashtekar_torsion_axial_current
        #    Torsion = clip((DealPrice-BuyPr)/(Spread+1e-6)-0.5, -0.5,0.5)
        #    sum( Torsion * TickVolRatio )
        df["_Torsion"] = np.clip(
            (df["DealPrice"] - df["BuyPr"]) / (df["_Spread"].replace(0, 1e-6) + 1e-6) - 0.5,
            -0.5, 0.5,
        )
        df["_axial"] = df["_Torsion"] * df["_phy_VR"]
        day_agg["raw_phy_ashtekar_torsion_axial_current"] = (
            df.groupby(["StockId", "Date_int"])["_axial"].sum()
            .reindex(day_agg.index).fillna(0).values
        )

        # 13. f_phy_chiral_parity_violating_flux
        #    sum( (dSpread/RefPr) * TickVolRatio * PrFlagSign )
        df["_chiral"] = (df["_dSpread"] / df["_phy_ref"]) * df["_phy_VR"] * df["_phy_FS"]
        day_agg["raw_phy_chiral_parity_violating_flux"] = (
            df.groupby(["StockId", "Date_int"])["_chiral"].sum()
            .reindex(day_agg.index).fillna(0).values
        )

        # 14. f_phy_parity_transformation_invariant
        #    sum( (dSpread/RefPr) * PrFlagSign * TickVolRatio * 1000 )
        df["_parity"] = (
            (df["_dSpread"] / df["_phy_ref"]) * df["_phy_FS"] * df["_phy_VR"] * 1000
        )
        day_agg["raw_phy_parity_transformation_invariant"] = (
            df.groupby(["StockId", "Date_int"])["_parity"].sum()
            .reindex(day_agg.index).fillna(0).values
        )

        # 15. f_phy_meromorphic_zero_dominance
        #    dt = clip(diff(DealTimeSecond), 0, 60)  (already sorted by TotalQty)
        #    sum( (dMidP/RefPr) * dt_cap )
        df["_dt_sec"] = np.clip(
            df.groupby(["StockId", "Date_int"])["DealTimeSecond"].diff().fillna(0),
            0, 60,
        )
        df["_MidRet"] = df["_dMidP"] / df["_phy_ref"]
        df["_mero"] = df["_MidRet"] * df["_dt_sec"]
        day_agg["raw_phy_meromorphic_zero_dominance"] = (
            df.groupby(["StockId", "Date_int"])["_mero"].sum()
            .reindex(day_agg.index).fillna(0).values
        )

        # 16. f_phy_scattering_vs_medium_reservoir_flux
        #    QuoteRet = |dBuyPr + dSellPr| / (2*RefPr)
        #    sum( TickVolRatio * PrFlagSign * QuoteRet * 1000 )
        df["_QuoteRet"] = np.abs(df["_dBuyPr"] + df["_dSellPr"]) / (2 * df["_phy_ref"])
        df["_scat"] = df["_phy_VR"] * df["_phy_FS"] * df["_QuoteRet"] * 1000
        day_agg["raw_phy_scattering_vs_medium_reservoir_flux"] = (
            df.groupby(["StockId", "Date_int"])["_scat"].sum()
            .reindex(day_agg.index).fillna(0).values
        )
    else:
        for _fn in [
            "ashtekar_torsion_axial_current",
            "chiral_parity_violating_flux",
            "parity_transformation_invariant",
            "meromorphic_zero_dominance",
            "scattering_vs_medium_reservoir_flux",
        ]:
            day_agg[f"raw_phy_{_fn}"] = 0.0

    # --- Cleanup all temporary columns ---
    _phy_drop = [
        "_phy_ref", "_phy_dP", "_phy_TR", "_phy_avg_vol", "_phy_VR", "_phy_FS",
        "_cas", "_vac", "_wigner_mask", "_wigner", "_TickFlow", "_collin",
        "_TR_up", "_TR_dn", "_BuyVR", "_SellVR", "_TR_abs",
        "_Amt", "_VWAP_t", "_DevRet", "_TrapComp",
        "_PCh", "_BlkId", "_ConsecTicks", "_NormConsecTicks", "_frac",
        "_HHMM", "_NetBuyCnt",
        "_CumVol_P", "_PrevCumVol_P", "_DepthRatio", "_optical",
        "_IsRev", "_RevFreq", "_quench",
        "_Spread", "_MidP", "_dSpread", "_dMidP", "_dBuyPr", "_dSellPr",
        "_Torsion", "_axial", "_chiral", "_parity",
        "_dt_sec", "_MidRet", "_mero", "_QuoteRet", "_scat",
    ]
    df.drop(
        columns=[c for c in _phy_drop if c in df.columns],
        inplace=True,
        errors="ignore",
    )
    # === End Alpha v6 PHY Physics Features ===

    # =========================================================================
    # === Alpha v7: 23 New Physics Features (Batch 20260312) ==================
    # =========================================================================
    # NOTE: df is already sorted by [StockId, Date_int, TotalQty] at the top.
    #       All computations below only ADD new raw_ columns; nothing existing is changed.

    # --- Shared helpers for this block ---
    eps = 1e-10
    BIN50  = 50
    BIN30  = 30
    BIN20  = 20
    BIN10  = 10
    BIN100 = 100

    # Compute NetFlow (PrFlag==1 → buy, PrFlag==0 → sell, else 0)
    df["_nf"] = np.where(df["PrFlag"] == 1, df["DealCount"].astype(float),
                np.where(df["PrFlag"] == 0, -df["DealCount"].astype(float), 0.0))

    # ---- 1. f_mhd_magnetic_helicity ----------------------------------------
    # A = cumsum(NetFlow) within each day, B = tanh(ΔP * 10)
    df["_A_mhd"] = df.groupby(["StockId", "Date_int"])["_nf"].cumsum()
    df["_dP_mhd"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff().fillna(0)
    df["_B_mhd"]  = np.tanh(df["_dP_mhd"] * 10)
    df["_AB_mhd"] = df["_A_mhd"] * df["_B_mhd"]
    _mhd_sum = df.groupby(["StockId", "Date_int"])["_AB_mhd"].sum()
    _mhd_cnt = df.groupby(["StockId", "Date_int"])["_AB_mhd"].count()
    _mhd_maxA = df.groupby(["StockId", "Date_int"])["_A_mhd"].apply(lambda x: x.abs().max())
    _mhd_val = _mhd_sum / (_mhd_cnt * _mhd_maxA + eps)
    _mhd_val[_mhd_cnt < 30] = np.nan
    day_agg["raw_mhd_magnetic_helicity"] = _mhd_val.reindex(day_agg.index).values
    df.drop(columns=["_A_mhd", "_dP_mhd", "_B_mhd", "_AB_mhd"], inplace=True, errors="ignore")

    # ---- 2. f_mhd_beltrami_alignment ----------------------------------------
    # 50-tick bins, B = price_end - price_start, J = diff of large_net per bin, cosine sim
    df["_large_thr_b"] = df.groupby(["StockId", "Date_int"])["DealCount"].transform(lambda x: x.quantile(0.75))
    df["_lnf_b"] = np.where(df["DealCount"] >= df["_large_thr_b"], df["_nf"], 0.0)
    df["_tseq_b"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_b"] = df["_tseq_b"] // BIN50
    _bagg_b = df.groupby(["StockId", "Date_int", "_bid_b"]).agg(
        _pfirst=("DealPrice", "first"),
        _plast=("DealPrice", "last"),
        _lnet=("_lnf_b", "sum")
    ).reset_index()
    _bagg_b["_B"] = _bagg_b["_plast"] - _bagg_b["_pfirst"]
    _bagg_b["_J"] = _bagg_b.groupby(["StockId", "Date_int"])["_lnet"].diff().fillna(0)
    def _beltrami_cosine(g):
        B, J = g["_B"].values, g["_J"].values
        n = len(B)
        if n < 5:
            return np.nan
        dot = np.sum(B * J)
        nb  = np.sqrt(np.sum(B**2)) + eps
        nj  = np.sqrt(np.sum(J**2)) + eps
        return float(np.clip(dot / (nb * nj), -1, 1))
    _bel = _bagg_b.groupby(["StockId", "Date_int"]).apply(_beltrami_cosine)
    day_agg["raw_mhd_beltrami_alignment"] = _bel.reindex(day_agg.index).values
    df.drop(columns=["_large_thr_b", "_lnf_b", "_tseq_b", "_bid_b"], inplace=True, errors="ignore")

    # ---- 3. f_mhd_vortex_asymmetric_contraction ----------------------------
    # log(det_cov_up / det_cov_dn) in (dp_norm, vol_norm) space
    df["_dP_v"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff()
    df["_dp_n"] = df.groupby(["StockId", "Date_int"])["_dP_v"].transform(
        lambda x: x / (x.abs().median() + eps))
    df["_vn_v"] = df.groupby(["StockId", "Date_int"])["DealCount"].transform(
        lambda x: x / (x.median() + eps))
    def _vortex(g):
        up = g[g["_dP_v"] > 0]
        dn = g[g["_dP_v"] < 0]
        if len(up) < 30 or len(dn) < 30:
            return np.nan
        def _cdet(sub):
            x, y = sub["_dp_n"].values, sub["_vn_v"].values
            cm = np.cov(x, y)
            return np.linalg.det(cm) + eps
        du, dd = _cdet(up), _cdet(dn)
        if du <= 0 or dd <= 0:
            return 0.0
        return float(np.log(du / dd))
    _vort = df.dropna(subset=["_dP_v"]).groupby(["StockId", "Date_int"]).apply(_vortex)
    day_agg["raw_mhd_vortex_asymmetric_contraction"] = _vort.reindex(day_agg.index).values
    df.drop(columns=["_dP_v", "_dp_n", "_vn_v"], inplace=True, errors="ignore")

    # ---- 4. f_pvg_negative_temperature (raw part only; confinement in calculate) ----
    # corr(|Price - Open|, NetFlow)  per day  →  temperature_sign
    _open_p = df.groupby(["StockId", "Date_int"])["DealPrice"].first()
    df["_open_map"] = df.set_index(["StockId", "Date_int"]).index.map(_open_p)
    df["_E_pvg"] = (df["DealPrice"] - df["_open_map"]).abs()
    def _neg_cov(g):
        if len(g) < 30:
            return np.nan
        e, nf = g["_E_pvg"].values, g["_nf"].values
        es, ns = e.std(), nf.std()
        if es < eps or ns < eps:
            return 0.0
        return float(np.corrcoef(e, nf)[0, 1])
    _neg_temp = df.groupby(["StockId", "Date_int"]).apply(_neg_cov)
    # Store daily_vol for confinement calc in feature script
    day_agg["raw_pvg_negative_temperature_corr"] = _neg_temp.reindex(day_agg.index).values
    # total_vol is already in day_agg as 'total_vol'
    df.drop(columns=["_open_map", "_E_pvg"], inplace=True, errors="ignore")

    # ---- 5. f_pvg_spin_glass_susceptibility --------------------------------
    # 100-tick bins, spin = sign(sum NetFlow), autocorr_sum * mean_spin
    df["_tseq_sg"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_sg"]  = df["_tseq_sg"] // BIN100
    _sg_bin = df.groupby(["StockId", "Date_int", "_bid_sg"])["_nf"].sum().reset_index()
    _sg_bin["_spin"] = np.sign(_sg_bin["_nf"])
    def _suscept(g):
        spins = g["_spin"].values
        n = len(spins)
        if n < 10:
            return np.nan
        max_lag = min(5, n // 2)
        ac_sum = 0.0
        for lag in range(1, max_lag + 1):
            c = np.corrcoef(spins[:-lag], spins[lag:])[0, 1]
            if np.isnan(c):
                c = 0.0
            ac_sum += c
        return float(ac_sum * spins.mean())
    _susc = _sg_bin.groupby(["StockId", "Date_int"]).apply(_suscept)
    day_agg["raw_pvg_spin_glass_susceptibility"] = _susc.reindex(day_agg.index).values
    df.drop(columns=["_tseq_sg", "_bid_sg"], inplace=True, errors="ignore")

    # ---- 6. f_pvg_high_energy_reference_shift ------------------------------
    # E0 = large-lot VWAP (≥75th pct); log(vol_above_E0 / vol_below_E0)
    df["_lthre_e"] = df.groupby(["StockId", "Date_int"])["DealCount"].transform(
        lambda x: x.quantile(0.75))
    _large_mask_e = df["DealCount"] >= df["_lthre_e"]
    _E0 = (df[_large_mask_e].groupby(["StockId", "Date_int"])
           .apply(lambda g: (g["DealPrice"] * g["DealCount"]).sum() / (g["DealCount"].sum() + eps))
           .rename("_E0"))
    df["_E0_val"] = df.set_index(["StockId", "Date_int"]).index.map(_E0)
    df["_above_e"] = np.where(df["DealPrice"] > df["_E0_val"], df["DealCount"], 0)
    df["_below_e"] = np.where(df["DealPrice"] <= df["_E0_val"], df["DealCount"], 0)
    _ev_agg = df.groupby(["StockId", "Date_int"]).agg(
        _va=("_above_e", "sum"), _vb=("_below_e", "sum"), _cnt_e=("DealPrice", "count"))
    _ev_agg["_raw"] = np.log((_ev_agg["_va"].astype(float) + eps) /
                              (_ev_agg["_vb"].astype(float) + eps))
    _ev_agg.loc[_ev_agg["_cnt_e"] < 30, "_raw"] = np.nan
    day_agg["raw_pvg_high_energy_reference_shift"] = _ev_agg["_raw"].reindex(day_agg.index).values
    df.drop(columns=["_lthre_e", "_E0_val", "_above_e", "_below_e"], inplace=True, errors="ignore")

    # ---- 7. f_lrt_fd_asymmetry ----------------------------------------------
    # dp_per_vol = ΔP / DealCount; R = |mean| / std per up/down group
    df["_dP_lrt"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff()
    df["_dpv"]    = df["_dP_lrt"] / (df["DealCount"] + eps)
    _up_m  = df.dropna(subset=["_dP_lrt"]).loc[df["_dP_lrt"] > 0].groupby(["StockId", "Date_int"])["_dpv"].agg(["mean", "std", "count"])
    _dn_m  = df.dropna(subset=["_dP_lrt"]).loc[df["_dP_lrt"] < 0].groupby(["StockId", "Date_int"])["_dpv"].agg(["mean", "std", "count"])
    _up_m.columns = ["up_mean", "up_std", "up_cnt"]
    _dn_m.columns = ["dn_mean", "dn_std", "dn_cnt"]
    _fd = _up_m.join(_dn_m, how="outer")
    _fd["_Ru"] = _fd["up_mean"].abs() / (_fd["up_std"].abs() + eps)
    _fd["_Rd"] = _fd["dn_mean"].abs() / (_fd["dn_std"].abs() + eps)
    _fd["_raw"] = (_fd["_Ru"] - _fd["_Rd"]) / (_fd["_Ru"] + _fd["_Rd"] + eps)
    _sparse_fd = (_fd["up_cnt"].fillna(0) < 10) | (_fd["dn_cnt"].fillna(0) < 10)
    _fd.loc[_sparse_fd, "_raw"] = np.nan
    day_agg["raw_lrt_fd_asymmetry"] = _fd["_raw"].reindex(day_agg.index).values
    df.drop(columns=["_dP_lrt", "_dpv"], inplace=True, errors="ignore")

    # ---- 8. f_lrt_kubo_response_polarization --------------------------------
    # Large orders (≥90th pct), response = price shift 5 ticks ahead (intraday, no lookahead)
    RESP_W = 5
    df["_lthre_k"] = df.groupby(["StockId", "Date_int"])["DealCount"].transform(
        lambda x: x.quantile(0.90))
    df["_is_lg_k"] = df["DealCount"] >= df["_lthre_k"]
    df["_fp_k"]   = df.groupby(["StockId", "Date_int"])["DealPrice"].shift(-RESP_W)
    df["_pr_k"]   = df["_fp_k"] - df["DealPrice"]
    _large_k = df[df["_is_lg_k"] & df["_pr_k"].notna()].copy()
    _buy_k  = _large_k[_large_k["PrFlag"] == 1]
    _sell_k = _large_k[_large_k["PrFlag"] == 0]
    _br = _buy_k.groupby(["StockId", "Date_int"]).agg(
        _mr=("_pr_k", "mean"), _mv=("DealCount", "mean"), _cnt=("_pr_k", "count"))
    _sr = _sell_k.groupby(["StockId", "Date_int"]).agg(
        _mr=("_pr_k", "mean"), _mv=("DealCount", "mean"), _cnt=("_pr_k", "count"))
    _cb = _br["_mr"] / (_br["_mv"] + eps)
    _cs = _sr["_mr"] / (_sr["_mv"] + eps)
    _kub = pd.DataFrame({"_cb": _cb, "_cs": _cs, "_cntb": _br["_cnt"], "_cnts": _sr["_cnt"]})
    _kub["_raw"] = (_kub["_cb"] + _kub["_cs"]) / (_kub["_cb"].abs() + _kub["_cs"].abs() + eps)
    _kub.loc[(_kub["_cntb"].fillna(0) < 3) | (_kub["_cnts"].fillna(0) < 3), "_raw"] = np.nan
    day_agg["raw_lrt_kubo_response_polarization"] = _kub["_raw"].reindex(day_agg.index).values
    df.drop(columns=["_lthre_k", "_is_lg_k", "_fp_k", "_pr_k"], inplace=True, errors="ignore")

    # ---- 9. f_lrt_conjugate_coskewness -------------------------------------
    # 50-tick bins: Co-Skewness(bin_vol, bin_vwap_dev)
    df["_tseq_cs"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_cs"]  = df["_tseq_cs"] // BIN50
    df["_pv_cs"]   = df["DealPrice"] * df["DealCount"]
    _cs_bin = df.groupby(["StockId", "Date_int", "_bid_cs"]).agg(
        _vol=("DealCount", "sum"),
        _pv=("_pv_cs", "sum"),
        _cnt=("DealCount", "count")
    )
    _cs_bin["_vwap"] = _cs_bin["_pv"] / (_cs_bin["_vol"] + eps)
    _dvwap = (df.groupby(["StockId", "Date_int"])
              .apply(lambda g: np.average(g["DealPrice"], weights=g["DealCount"]))
              .rename("_dvwap"))
    _cs_bin = _cs_bin.reset_index()
    _cs_bin = _cs_bin.merge(_dvwap.reset_index().rename(columns={0: "_dvwap"}),
                             on=["StockId", "Date_int"], how="left")
    _cs_bin["_vwap_dev"] = (_cs_bin["_vwap"] - _cs_bin["_dvwap"]) / (_cs_bin["_dvwap"] + eps)
    def _coskew(g):
        if len(g) < 10:
            return np.nan
        v = g["_vol"].values.astype(float)
        p = g["_vwap_dev"].values
        vc, pc = v - v.mean(), p - p.mean()
        sv, sp = v.std(), p.std()
        if sv < eps or sp < eps:
            return 0.0
        return float(np.mean(vc**2 * pc) / (sv**2 * sp + eps))
    _csk = _cs_bin.groupby(["StockId", "Date_int"]).apply(_coskew)
    day_agg["raw_lrt_conjugate_coskewness"] = _csk.reindex(day_agg.index).values
    df.drop(columns=["_tseq_cs", "_bid_cs", "_pv_cs"], inplace=True, errors="ignore")

    # ---- 10. f_crack_creep_to_jump ------------------------------------------
    # 30-tick bins, up_eat / down_eat 2nd diff (acceleration difference)
    if "BuyPr" in df.columns and "SellPr" in df.columns:
        df["_up_eat"] = np.where(
            (df["PrFlag"] == 1) & (df["DealPrice"] >= df["SellPr"]),
            df["DealCount"].astype(float), 0.0)
        df["_dn_eat"] = np.where(
            (df["PrFlag"] == 0) & (df["DealPrice"] <= df["BuyPr"]),
            df["DealCount"].astype(float), 0.0)
        df["_tseq_ck"] = df.groupby(["StockId", "Date_int"]).cumcount()
        df["_bid_ck"]  = df["_tseq_ck"] // BIN30
        _ck_bin = df.groupby(["StockId", "Date_int", "_bid_ck"]).agg(
            _up=("_up_eat", "sum"), _dn=("_dn_eat", "sum"))
        _ck_bin["_up_acc"] = _ck_bin.groupby(["StockId", "Date_int"])["_up"].diff().diff().fillna(0)
        _ck_bin["_dn_acc"] = _ck_bin.groupby(["StockId", "Date_int"])["_dn"].diff().diff().fillna(0)
        _ck_day = _ck_bin.groupby(["StockId", "Date_int"]).agg(
            _mua=("_up_acc", "mean"), _mda=("_dn_acc", "mean"), _nc=("_up_acc", "count"))
        _ck_day["_raw"] = (_ck_day["_mua"] - _ck_day["_mda"]) / (
            _ck_day["_mua"].abs() + _ck_day["_mda"].abs() + eps)
        _ck_day.loc[_ck_day["_nc"] < 5, "_raw"] = np.nan
        day_agg["raw_crack_creep_to_jump"] = _ck_day["_raw"].reindex(day_agg.index).values
        df.drop(columns=["_up_eat", "_dn_eat", "_tseq_ck", "_bid_ck"], inplace=True, errors="ignore")
    else:
        day_agg["raw_crack_creep_to_jump"] = np.nan

    # ---- 11. f_crack_time_reversal_action -----------------------------------
    # S_fwd = Σ|flow|×|ΔP|, S_rev = Σ|flow_rev|×|ΔP|, × sign(close-open)
    df["_dP_tr"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff().fillna(0)
    df["_af_tr"] = df["_nf"].abs()
    df["_adp_tr"] = df["_dP_tr"].abs()
    def _tra(g):
        n = len(g)
        if n < 30:
            return np.nan
        af  = g["_af_tr"].values
        adp = g["_adp_tr"].values
        s_fwd = np.sum(af * adp)
        s_rev = np.sum(af[::-1] * adp)
        ad = (s_fwd - s_rev) / (s_fwd + s_rev + eps)
        op = g["DealPrice"].iloc[0]
        cl = g["DealPrice"].iloc[-1]
        return float(ad * (np.sign(cl - op) if cl != op else 1e-5))
    _tra_v = df.groupby(["StockId", "Date_int"]).apply(_tra)
    day_agg["raw_crack_time_reversal_action"] = _tra_v.reindex(day_agg.index).values
    df.drop(columns=["_dP_tr", "_af_tr", "_adp_tr"], inplace=True, errors="ignore")

    # ---- 12. f_crack_liquidity_collapse -------------------------------------
    # 50-tick bins, BuyPr/SellPr median per bin, changes → advance/retreat
    if "BuyPr" in df.columns and "SellPr" in df.columns:
        df["_tseq_lc"] = df.groupby(["StockId", "Date_int"]).cumcount()
        df["_bid_lc"]  = df["_tseq_lc"] // BIN50
        _lc_bin = df.groupby(["StockId", "Date_int", "_bid_lc"]).agg(
            _bp=("BuyPr", "median"), _sp=("SellPr", "median"), _rp=("DealPrice", "median")).reset_index()
        _lc_bin["_bch"] = _lc_bin.groupby(["StockId", "Date_int"])["_bp"].diff().fillna(0)
        _lc_bin["_sch"] = _lc_bin.groupby(["StockId", "Date_int"])["_sp"].diff().fillna(0)
        _lc_bin["_bchn"] = _lc_bin["_bch"] / (_lc_bin["_rp"] + eps)
        _lc_bin["_schn"] = _lc_bin["_sch"] / (_lc_bin["_rp"] + eps)
        _lc_day = _lc_bin.groupby(["StockId", "Date_int"]).agg(
            _sa=("_schn", lambda x: x[x > 0].sum()),
            _br=("_bchn", lambda x: x[x < 0].sum()),
            _sr=("_schn", lambda x: x[x < 0].sum()),
            _ba=("_bchn", lambda x: x[x > 0].sum()),
            _nc=("_bid_lc", "count")
        )
        _lc_day["_raw"] = ((_lc_day["_sa"] + _lc_day["_ba"]) /
                            (_lc_day["_sa"].abs() + _lc_day["_ba"].abs() +
                             _lc_day["_sr"].abs() + _lc_day["_br"].abs() + eps))
        _lc_day.loc[_lc_day["_nc"] < 5, "_raw"] = np.nan
        day_agg["raw_crack_liquidity_collapse"] = _lc_day["_raw"].reindex(day_agg.index).values
        df.drop(columns=["_tseq_lc", "_bid_lc"], inplace=True, errors="ignore")
    else:
        day_agg["raw_crack_liquidity_collapse"] = np.nan

    # ---- 13. f_sindy_drift_polarity -----------------------------------------
    # 30-tick bins, OLS t-stat of intraday return drift (α / std_ret)
    df["_tseq_sd"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_sd"]  = df["_tseq_sd"] // BIN30
    _sd_bin = df.groupby(["StockId", "Date_int", "_bid_sd"]).agg(
        _price=("DealPrice", "last")).reset_index()
    _sd_bin["_dp"] = _sd_bin.groupby(["StockId", "Date_int"])["_price"].diff()
    _sd_bin["_plag"] = _sd_bin.groupby(["StockId", "Date_int"])["_price"].shift(1)
    _sd_bin = _sd_bin.dropna(subset=["_dp", "_plag"])
    _sd_bin["_ret"] = _sd_bin["_dp"] / (_sd_bin["_plag"] + eps)
    def _sindy_dp(g):
        if len(g) < 10:
            return np.nan
        ret = g["_ret"].values
        s = ret.std()
        if s < eps:
            return 0.0
        return float(ret.mean() / (s + eps))
    _sdp = _sd_bin.groupby(["StockId", "Date_int"]).apply(_sindy_dp)
    day_agg["raw_sindy_drift_polarity"] = _sdp.reindex(day_agg.index).values
    df.drop(columns=["_tseq_sd", "_bid_sd"], inplace=True, errors="ignore")

    # ---- 14. f_sindy_residual_asymmetry -------------------------------------
    # 20-tick bins, skew of EWMA prediction residuals
    EWM_SPAN_SR = 20
    df["_tseq_sr"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_sr"]  = df["_tseq_sr"] // BIN20
    _sr_bin = df.groupby(["StockId", "Date_int", "_bid_sr"]).agg(
        _price=("DealPrice", "last")).reset_index()
    _sr_bin["_dp"] = _sr_bin.groupby(["StockId", "Date_int"])["_price"].diff()
    def _sindy_res_skew(g):
        dp = g["_dp"].dropna().values
        if len(dp) < 20:
            return np.nan
        dp_s = pd.Series(dp)
        ewma  = dp_s.ewm(span=EWM_SPAN_SR, min_periods=3).mean().shift(1)
        resid = (dp_s - ewma).dropna()
        if len(resid) < 10:
            return np.nan
        return float(resid.skew())
    _srsk = _sr_bin.groupby(["StockId", "Date_int"]).apply(_sindy_res_skew)
    day_agg["raw_sindy_residual_asymmetry"] = _srsk.reindex(day_agg.index).values
    df.drop(columns=["_tseq_sr", "_bid_sr"], inplace=True, errors="ignore")

    # ---- 15. f_sindy_phase_transition_gain ----------------------------------
    # KL(P_first || P_second) for first/second half-by-cumvol DP distribution × direction
    df["_dP_pt"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff()
    df_pt = df.dropna(subset=["_dP_pt"]).copy()
    df_pt["_cumv_pt"]  = df_pt.groupby(["StockId", "Date_int"])["DealCount"].cumsum()
    df_pt["_totv_pt"]  = df_pt.groupby(["StockId", "Date_int"])["DealCount"].transform("sum")
    df_pt["_half_pt"]  = np.where(df_pt["_cumv_pt"] <= df_pt["_totv_pt"] / 2, 0, 1)
    def _kl_gain(g):
        first  = g[g["_half_pt"] == 0]["_dP_pt"].values
        second = g[g["_half_pt"] == 1]["_dP_pt"].values
        if len(first) < 20 or len(second) < 20:
            return np.nan
        mu1, s1 = first.mean(), first.std() + eps
        mu2, s2 = second.mean(), second.std() + eps
        kl = np.log(s2 / s1) + (s1**2 + (mu1 - mu2)**2) / (2 * s2**2) - 0.5
        direction = np.sign(mu2 - mu1)
        if direction == 0:
            direction = 1e-5
        return float(kl * direction)
    _ptg = df_pt.groupby(["StockId", "Date_int"]).apply(_kl_gain)
    day_agg["raw_sindy_phase_transition_gain"] = _ptg.reindex(day_agg.index).values
    df.drop(columns=["_dP_pt"], inplace=True, errors="ignore")

    # ---- 16. f_qm_coarse_graining_loss_bias ---------------------------------
    # 5-min bars (using DealTimeSecond), info loss asymmetry
    if "DealTimeSecond" in df.columns:
        df["_5min_bin"] = df["DealTimeSecond"] // 300
        df["_dP_cg"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff().fillna(0)
        df["_dp_up"] = df["_dP_cg"].clip(lower=0)
        df["_dp_dn"] = df["_dP_cg"].clip(upper=0)
        _cg_bar = df.groupby(["StockId", "Date_int", "_5min_bin"]).agg(
            _macro=("DealPrice", lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0),
            _mup=("_dp_up", "sum"),
            _mdn=("_dp_dn", "sum"),
            _nc=("_dP_cg", "count")
        ).reset_index()
        _cg_bar["_ul"] = _cg_bar["_mup"] - _cg_bar["_macro"].clip(lower=0)
        _cg_bar["_dl"] = _cg_bar["_mdn"].abs() - (-_cg_bar["_macro"]).clip(lower=0)
        _cg_day = _cg_bar.groupby(["StockId", "Date_int"]).agg(
            _uls=("_ul", "sum"), _dls=("_dl", "sum"),
            _td=("_macro", lambda x: x.abs().sum()), _nb=("_5min_bin", "count"))
        _cg_day["_raw"] = (_cg_day["_dls"] - _cg_day["_uls"]) / (_cg_day["_td"] + eps)
        _cg_day.loc[_cg_day["_nb"] < 5, "_raw"] = np.nan
        day_agg["raw_qm_coarse_graining_loss_bias"] = _cg_day["_raw"].reindex(day_agg.index).values
        df.drop(columns=["_5min_bin", "_dP_cg", "_dp_up", "_dp_dn"], inplace=True, errors="ignore")
    else:
        day_agg["raw_qm_coarse_graining_loss_bias"] = np.nan

    # ---- 17. f_qm_density_offdiag_relaxation --------------------------------
    # 50-tick bins, cross-corr decay of large/small flow × sign_lag1
    df["_lthre_qd"] = df.groupby(["StockId", "Date_int"])["DealCount"].transform(
        lambda x: x.quantile(0.75))
    df["_lg_qd"] = np.where(df["DealCount"] >= df["_lthre_qd"], df["_nf"], 0.0)
    df["_sm_qd"] = np.where(df["DealCount"] <  df["_lthre_qd"], df["_nf"], 0.0)
    df["_tseq_qd"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_qd"]  = df["_tseq_qd"] // BIN50
    _qd_bin = df.groupby(["StockId", "Date_int", "_bid_qd"]).agg(
        _lf=("_lg_qd", "sum"), _sf=("_sm_qd", "sum")).reset_index()
    def _coher_decay(g):
        n = len(g)
        if n < 10:
            return np.nan
        large = g["_lf"].values
        small = g["_sf"].values
        max_lag = min(4, n // 3)
        corrs = []
        for lag in range(max_lag + 1):
            if lag == 0:
                c = np.corrcoef(large, small)[0, 1]
            else:
                if len(large[:-lag]) < 3:
                    break
                c = np.corrcoef(large[:-lag], small[lag:])[0, 1]
            corrs.append(0.0 if np.isnan(c) else c)
        if len(corrs) < 3:
            return np.nan
        log_c, taus = [], []
        for i, c in enumerate(corrs):
            if abs(c) > eps:
                log_c.append(np.log(abs(c)))
                taus.append(i)
        if len(log_c) < 2:
            return 0.0
        slope = np.polyfit(np.array(taus, float), np.array(log_c), 1)[0]
        sign1 = np.sign(corrs[1]) if len(corrs) > 1 else 0
        return float((-slope) * sign1)
    _qdc = _qd_bin.groupby(["StockId", "Date_int"]).apply(_coher_decay)
    day_agg["raw_qm_density_offdiag_relaxation"] = _qdc.reindex(day_agg.index).values
    df.drop(columns=["_lthre_qd", "_lg_qd", "_sm_qd", "_tseq_qd", "_bid_qd"],
            inplace=True, errors="ignore")

    # ---- 18. f_pol_effective_mass_asymmetry ---------------------------------
    # Mass_up = Σvol(up) / Σ|ΔP(up)|, Mass_down similarly; (Mass_dn - Mass_up) / sum
    df["_dP_pm"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff()
    df_pm = df.dropna(subset=["_dP_pm"]).copy()
    df_pm["_adp_pm"] = df_pm["_dP_pm"].abs()
    _up_pmu = df_pm[df_pm["_dP_pm"] > 0].groupby(["StockId", "Date_int"])["DealCount"].sum()
    _up_pmd = df_pm[df_pm["_dP_pm"] > 0].groupby(["StockId", "Date_int"])["_adp_pm"].sum()
    _dn_pmu = df_pm[df_pm["_dP_pm"] < 0].groupby(["StockId", "Date_int"])["DealCount"].sum()
    _dn_pmd = df_pm[df_pm["_dP_pm"] < 0].groupby(["StockId", "Date_int"])["_adp_pm"].sum()
    _pm_df = pd.DataFrame({"_uv": _up_pmu, "_ud": _up_pmd, "_dv": _dn_pmu, "_dd": _dn_pmd})
    _pm_df["_mu"] = _pm_df["_uv"] / (_pm_df["_ud"] + eps)
    _pm_df["_md"] = _pm_df["_dv"] / (_pm_df["_dd"] + eps)
    _pm_df["_raw"] = (_pm_df["_md"] - _pm_df["_mu"]) / (_pm_df["_md"] + _pm_df["_mu"] + eps)
    _pm_df.loc[(_pm_df["_ud"] < eps) | (_pm_df["_dd"] < eps), "_raw"] = np.nan
    day_agg["raw_pol_effective_mass_asymmetry"] = _pm_df["_raw"].reindex(day_agg.index).values
    df.drop(columns=["_dP_pm"], inplace=True, errors="ignore")

    # ---- 19. f_pol_spectral_centroid_shift (raw SC × direction) -------------
    # 20-tick bins, FFT of momentum series, SC × sign(close-open)
    df["_tseq_psc"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_psc"]  = df["_tseq_psc"] // BIN20
    _psc_bin = df.groupby(["StockId", "Date_int", "_bid_psc"]).agg(
        _pl=("DealPrice", "last"), _vs=("DealCount", "sum")).reset_index()
    _psc_bin["_dp"] = _psc_bin.groupby(["StockId", "Date_int"])["_pl"].diff().fillna(0)
    _psc_bin["_mom"] = _psc_bin["_dp"] * _psc_bin["_vs"]
    def _spectral(g):
        x = g["_mom"].values
        if len(x) < 16:
            return np.nan
        x = x - x.mean()
        fft_v = np.fft.rfft(x)
        power = np.abs(fft_v)**2
        freqs = np.fft.rfftfreq(len(x))
        power, freqs = power[1:], freqs[1:]
        if power.sum() < eps:
            return 0.0
        sc = np.sum(freqs * power) / (np.sum(power) + eps)
        op = g["_pl"].iloc[0]
        cl = g["_pl"].iloc[-1]
        d  = np.sign(cl - op) if cl != op else 1e-5
        return float(sc * d)
    _psc_v = _psc_bin.groupby(["StockId", "Date_int"]).apply(_spectral)
    day_agg["raw_pol_spectral_centroid_shift"] = _psc_v.reindex(day_agg.index).values
    df.drop(columns=["_tseq_psc", "_bid_psc"], inplace=True, errors="ignore")

    # ---- 20. f_pol_phonon_drag_coefficient ----------------------------------
    # Large orders, track next LOOK_AFTER=10 small ticks: follow_rate × log(drag) × direction
    LOOK_AFTER_PD = 10
    df["_lthre_pd"] = df.groupby(["StockId", "Date_int"])["DealCount"].transform(
        lambda x: x.quantile(0.90))
    df["_is_lg_pd"] = df["DealCount"] >= df["_lthre_pd"]
    df["_is_sm_pd"] = ~df["_is_lg_pd"]
    df["_dP_pd"]   = df.groupby(["StockId", "Date_int"])["DealPrice"].diff().fillna(0)
    df["_sgn_pd"]  = np.sign(df["_dP_pd"])
    df["_ldir_pd"] = np.where(df["_is_lg_pd"],
                              np.where(df["PrFlag"] == 1, 1.0, -1.0), np.nan)
    def _phonon(date_g):
        large_idx   = np.where(date_g["_is_lg_pd"].values)[0]
        if len(large_idx) < 3:
            return np.nan
        normal_sv = date_g.loc[date_g["_is_sm_pd"], "DealCount"].mean()
        if np.isnan(normal_sv) or normal_sv < eps:
            return np.nan
        sgn_arr   = date_g["_sgn_pd"].values
        vol_arr   = date_g["DealCount"].values.astype(float)
        is_sm_arr = date_g["_is_sm_pd"].values
        ldir_arr  = date_g["_ldir_pd"].values
        n = len(date_g)
        fv = []
        for li in large_idx:
            if li + LOOK_AFTER_PD >= n:
                continue
            big_dir = ldir_arr[li]
            if np.isnan(big_dir):
                continue
            after = np.arange(li + 1, min(li + 1 + LOOK_AFTER_PD * 3, n))
            sm_after = after[is_sm_arr[after]][:LOOK_AFTER_PD]
            if len(sm_after) < 3:
                continue
            follow = np.mean(sgn_arr[sm_after] == big_dir)
            drag   = vol_arr[sm_after].mean() / (normal_sv + eps)
            fv.append(follow * np.log1p(drag) * big_dir)
        if len(fv) < 3:
            return np.nan
        return float(np.mean(fv))
    _phon = df.groupby(["StockId", "Date_int"]).apply(_phonon)
    day_agg["raw_pol_phonon_drag_coefficient"] = _phon.reindex(day_agg.index).values
    df.drop(columns=["_lthre_pd", "_is_lg_pd", "_is_sm_pd", "_dP_pd", "_sgn_pd", "_ldir_pd"],
            inplace=True, errors="ignore")

    # ---- 21. f_geo_lyapunov_drift_vector ------------------------------------
    # 10-tick bins, ddp = 2nd diff of dp per bin, EWMA(ddp) / std(dp)
    EWM_SPAN_LY = 50
    df["_tseq_ly"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_ly"]  = df["_tseq_ly"] // BIN10
    _ly_bin = df.groupby(["StockId", "Date_int", "_bid_ly"]).agg(
        _price=("DealPrice", "last"), _vol=("DealCount", "sum")).reset_index()
    _ly_bin["_dp"]  = _ly_bin.groupby(["StockId", "Date_int"])["_price"].diff()
    _ly_bin["_dv"]  = _ly_bin.groupby(["StockId", "Date_int"])["_vol"].diff()
    _ly_bin = _ly_bin.dropna(subset=["_dp", "_dv"])
    _ly_bin["_ddp"] = _ly_bin.groupby(["StockId", "Date_int"])["_dp"].diff().fillna(0)
    def _lyapunov(g):
        ddp = g["_ddp"].values
        dp  = g["_dp"].values
        n   = len(ddp)
        if n < 20:
            return np.nan
        alpha = 2.0 / (EWM_SPAN_LY + 1)
        ewma  = 0.0
        for v in ddp:
            ewma = alpha * v + (1 - alpha) * ewma
        std_dp = dp.std()
        if std_dp < eps:
            return 0.0
        return float(ewma / std_dp)
    _lyap = _ly_bin.groupby(["StockId", "Date_int"]).apply(_lyapunov)
    day_agg["raw_geo_lyapunov_drift_vector"] = _lyap.reindex(day_agg.index).values
    df.drop(columns=["_tseq_ly", "_bid_ly"], inplace=True, errors="ignore")

    # ---- 22. f_geo_entropy_directed_work ------------------------------------
    # 50-tick bins, F = (BuyVol-SellVol)/(BuyVol+SellVol), dx = close-open, W=ΣF×dx/norm
    df["_bv_gw"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
    df["_sv_gw"] = np.where(df["PrFlag"] == 0, df["DealCount"], 0)
    df["_tseq_gw"] = df.groupby(["StockId", "Date_int"]).cumcount()
    df["_bid_gw"]  = df["_tseq_gw"] // BIN50
    _gw_bin = df.groupby(["StockId", "Date_int", "_bid_gw"]).agg(
        _op=("DealPrice", "first"), _cl=("DealPrice", "last"),
        _bv=("_bv_gw", "sum"),  _sv=("_sv_gw", "sum"),
        _tv=("DealCount", "sum")).reset_index()
    _gw_bin["_F"]  = (_gw_bin["_bv"] - _gw_bin["_sv"]) / (_gw_bin["_bv"] + _gw_bin["_sv"] + eps)
    _gw_bin["_dx"] = _gw_bin["_cl"] - _gw_bin["_op"]
    _gw_bin["_W"]  = _gw_bin["_F"] * _gw_bin["_dx"]
    _gw_day = _gw_bin.groupby(["StockId", "Date_int"]).agg(
        _ws=("_W", "sum"),
        _ads=("_dx", lambda x: x.abs().sum()),
        _afm=("_F", lambda x: x.abs().mean()),
        _nc=("_bid_gw", "count")
    )
    _gw_day["_raw"] = _gw_day["_ws"] / (_gw_day["_ads"] * _gw_day["_afm"] + eps)
    _gw_day.loc[_gw_day["_nc"] < 5, "_raw"] = np.nan
    _gw_day.loc[_gw_day["_ads"] < eps, "_raw"] = 0.0
    day_agg["raw_geo_entropy_directed_work"] = _gw_day["_raw"].reindex(day_agg.index).values
    df.drop(columns=["_bv_gw", "_sv_gw", "_tseq_gw", "_bid_gw"], inplace=True, errors="ignore")

    # ---- 23. f_net_markov_flow_asymmetry ------------------------------------
    # vol-weighted up displacement vs down displacement normalized net flow
    df["_dP_mk"] = df.groupby(["StockId", "Date_int"])["DealPrice"].diff()
    df_mk = df.dropna(subset=["_dP_mk"]).copy()
    df_mk["_w"]  = df_mk["DealCount"].shift(-1).fillna(0)  # weight = next tick's volume
    df_mk["_up_mk"] = np.where(df_mk["_dP_mk"] > 0, df_mk["_w"] * df_mk["_dP_mk"], 0.0)
    df_mk["_dn_mk"] = np.where(df_mk["_dP_mk"] < 0, df_mk["_w"] * df_mk["_dP_mk"].abs(), 0.0)
    df_mk["_tot_mk"] = np.where(
        df_mk["_dP_mk"] != 0, df_mk["_w"] * df_mk["_dP_mk"].abs(), 0.0)
    _mk_day = df_mk.groupby(["StockId", "Date_int"]).agg(
        _vu=("_up_mk", "sum"), _vd=("_dn_mk", "sum"), _vt=("_tot_mk", "sum"), _cnt=("_dP_mk", "count"))
    _mk_day["_raw"] = np.where(
        _mk_day["_vt"] > eps,
        (_mk_day["_vu"] - _mk_day["_vd"]) / (_mk_day["_vt"] + eps), 0.0)
    _mk_day.loc[_mk_day["_cnt"] < 50, "_raw"] = np.nan
    day_agg["raw_net_markov_flow_asymmetry"] = _mk_day["_raw"].reindex(day_agg.index).values
    df.drop(columns=["_dP_mk"], inplace=True, errors="ignore")

    # --- Cleanup shared helpers ---
    df.drop(columns=["_nf"], inplace=True, errors="ignore")

    # === End Alpha v7: 23 New Physics Features ===

    # === Alpha v8: IT Group 3 Poisson Channel Features (3) ===

    # ---- F09: f_it_post_event_directional_impulse ----
    # Large trade event (>= 90th percentile) directional impulse after the event
    def _post_event_impulse(g):
        if len(g) < 30:
            return np.nan
        q90 = g["DealCount"].quantile(0.90)
        large_mask = g["DealCount"] >= q90
        if large_mask.sum() < 3:
            return np.nan
        
        prf = g["PrFlag"].values
        buy_responses = []
        sell_responses = []
        
        g_reset = g.reset_index(drop=True)
        for pos in range(len(g_reset)):
            if g_reset.iloc[pos]["DealCount"] < q90:
                continue
            event_flag = prf[pos]
            start = pos + 1
            end = min(pos + 11, len(g_reset))
            if end <= start:
                continue
            post_flags = prf[start:end]
            buy_ratio = (post_flags == 1).mean()
            
            if event_flag == 1:  # Buyer large event
                buy_responses.append(buy_ratio)
            else:  # Seller large event
                sell_responses.append(buy_ratio)
        
        mu_buy = np.mean(buy_responses) if buy_responses else 0.5
        mu_sell = np.mean(sell_responses) if sell_responses else 0.5
        return float(mu_buy - mu_sell)
    
    day_agg["raw_it_post_event_directional_impulse"] = df.groupby(["StockId", "Date_int"]).apply(_post_event_impulse).reindex(day_agg.index).values

    # ---- F10: f_it_large_trade_iat_asymmetry ----
    # Large trade IAT (Inter-Arrival Time) asymmetry
    def _iat_asymmetry(g):
        if len(g) < 30:
            return np.nan
        q90 = g["DealCount"].quantile(0.90)
        g_s = g.sort_values("DealTimeSecond").reset_index(drop=True)
        large = g_s[g_s["DealCount"] >= q90].copy()
        
        if len(large) < 6:
            return np.nan
        
        buy_large = large[large["PrFlag"] == 1]["DealTimeSecond"].values
        sell_large = large[large["PrFlag"] == 0]["DealTimeSecond"].values
        
        def iat_accel(times):
            if len(times) < 4:
                return 1.0
            times = np.sort(times)
            iats = np.diff(times).astype(float)
            if len(iats) < 2:
                return 1.0
            mid = len(iats) // 2
            first_half = iats[:mid].mean() + 1e-9
            second_half = iats[mid:].mean() + 1e-9
            return float(second_half / first_half)
        
        accel_buy = iat_accel(buy_large)
        accel_sell = iat_accel(sell_large)
        
        return float(accel_sell / (accel_buy + 1e-9) - 1.0)
    
    day_agg["raw_it_large_trade_iat_asymmetry"] = df.groupby(["StockId", "Date_int"]).apply(_iat_asymmetry).reindex(day_agg.index).values

    # ---- F11: f_it_broker_poisson_intensity_onesidedness ----
    # Poisson intensity onesidedness across 15-min windows
    def _poisson_onesidedness(g):
        if len(g) < 20:
            return np.nan
        q90 = g["DealCount"].quantile(0.90)
        large = g[g["DealCount"] >= q90].copy()
        
        if len(large) < 5:
            return np.nan
        
        # 14 windows of 15 minutes (09:00-13:30 = 270 minutes)
        START_SEC = 9 * 3600   # 32400
        n_windows = 14
        WINDOW_SEC = 15 * 60   # 900
        
        lambda_buy_list = []
        lambda_sell_list = []
        
        for w in range(n_windows):
            w_start = START_SEC + w * WINDOW_SEC
            w_end = w_start + WINDOW_SEC
            win = large[(large["DealTimeSecond"] >= w_start) & (large["DealTimeSecond"] < w_end)]
            n_buy = (win["PrFlag"] == 1).sum()
            n_sell = (win["PrFlag"] == 0).sum()
            lambda_buy_list.append(n_buy / WINDOW_SEC * 60)   # events per minute
            lambda_sell_list.append(n_sell / WINDOW_SEC * 60)
        
        lambda_buy = np.mean(lambda_buy_list)
        lambda_sell = np.mean(lambda_sell_list)
        return float((lambda_buy - lambda_sell) / (lambda_buy + lambda_sell + 1e-9))
    
    day_agg["raw_it_broker_poisson_intensity_onesidedness"] = df.groupby(["StockId", "Date_int"]).apply(_poisson_onesidedness).reindex(day_agg.index).values

    # === End Alpha v8: IT Group 3 Poisson Channel Features ===

    # === Alpha v17: Shared Intermediate Calculations (thresholds already computed at start) ===
    # These are used by multiple features to avoid redundant computation
    
    # _is_buy_vol: volume of active buys (PrFlag==1)
    df["_is_buy_vol"] = np.where(df["PrFlag"] == 1, df["DealCount"], 0)
    df["_is_sell_vol"] = np.where(df["PrFlag"] == 0, df["DealCount"], 0)
    
    # _small_buy, _small_sell: small trades (<= Small_Thresh)
    df["_small_buy"] = np.where(
        (df["PrFlag"] == 1) & (df["DealCount"] <= df["Small_Thresh"]),
        df["DealCount"], 0
    )
    df["_small_sell"] = np.where(
        (df["PrFlag"] == 0) & (df["DealCount"] <= df["Small_Thresh"]),
        df["DealCount"], 0
    )
    
    # _large_buy, _large_sell: large trades (>= Large_Thresh)
    df["_large_buy"] = np.where(
        (df["PrFlag"] == 1) & (df["DealCount"] >= df["Large_Thresh"]),
        df["DealCount"], 0
    )
    df["_large_sell"] = np.where(
        (df["PrFlag"] == 0) & (df["DealCount"] >= df["Large_Thresh"]),
        df["DealCount"], 0
    )
    
    # _spread: bid-ask spread (if available)
    if "BuyPr" in df.columns and "SellPr" in df.columns:
        df["_spread"] = (df["SellPr"] - df["BuyPr"]).clip(lower=0)
    else:
        df["_spread"] = 0
    
    # _pv: price * volume for VWAP calculations
    df["_pv"] = df["DealPrice"] * df["DealCount"]
    
    # === End Alpha v17 Shared Calculations ===

    # ============================================================================
    # === Alpha NLD: 16 New Nonlinear Dynamics Features ===
    # ============================================================================

    # ---- Helper functions for NLD features ----
    def _price_to_three_states(dp):
        states = np.ones(len(dp), dtype=int)
        states[dp > 0] = 0
        states[dp < 0] = 2
        return states

    def _build_transition_matrix(states, n_states, laplace_smooth=1.0):
        matrix = np.full((n_states, n_states), laplace_smooth)
        for i in range(len(states) - 1):
            s_from = int(states[i])
            s_to = int(states[i + 1])
            if 0 <= s_from < n_states and 0 <= s_to < n_states:
                matrix[s_from, s_to] += 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return matrix / row_sums

    def _stationary_distribution(matrix):
        try:
            n = matrix.shape[0]
            eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            v = eigenvectors[:, idx].real
            v = np.abs(v)
            s = v.sum()
            if s > 0:
                return v / s
            return np.ones(n) / n
        except Exception:
            n = matrix.shape[0]
            return np.ones(n) / n

    def _leading_eigenvalue_real(matrix):
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            idx = np.argmax(np.abs(eigenvalues))
            return eigenvalues[idx].real
        except Exception:
            return 0.0

    def _eigenvalue_gap(matrix):
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            abs_ev = np.sort(np.abs(eigenvalues.real))[::-1]
            if len(abs_ev) >= 2:
                return abs_ev[0] - abs_ev[1]
            return 0.0
        except Exception:
            return 0.0

    # ---- 1. f_nld_bifurcation_delay_eigen ----
    def _bifurcation_delay(day_df):
        prices = day_df['DealPrice'].values
        volumes = day_df['DealCount'].values.astype(float)
        time_sec = day_df['DealTimeSecond'].values
        if len(prices) < 30:
            return 0.0
        dp = np.diff(prices)
        dv = np.diff(volumes)
        t = time_sec[1:]
        bin_size = 900
        t_min = t.min()
        bins = (t - t_min) // bin_size
        unique_bins = np.unique(bins)
        if len(unique_bins) < 2:
            return 0.0
        eigen_gaps = []
        accels = []
        for b in unique_bins:
            mask = bins == b
            dp_bin = dp[mask]
            dv_bin = dv[mask]
            if len(dp_bin) < 5:
                continue
            dp_std = np.std(dp_bin) + 1e-10
            dv_std = np.std(dv_bin) + 1e-10
            data_mat = np.column_stack([dp_bin / dp_std, dv_bin / dv_std])
            try:
                cov = np.cov(data_mat.T)
                gap = _eigenvalue_gap(cov)
            except:
                gap = 0.0
            if len(dp_bin) >= 3:
                d2p = np.diff(dp_bin)
                accel = np.mean(d2p)
            else:
                accel = 0.0
            eigen_gaps.append(gap)
            accels.append(accel)
        if len(eigen_gaps) == 0:
            return 0.0
        eigen_gaps = np.array(eigen_gaps)
        accels = np.array(accels)
        min_gap_idx = np.argmin(eigen_gaps)
        asym = np.sign(prices[-1] - prices[0])
        return accels[min_gap_idx] * asym
    day_agg["raw_nld_bifurcation_delay_eigen"] = df.groupby(["StockId", "Date_int"]).apply(_bifurcation_delay).reindex(day_agg.index).values

    # ---- 2. f_nld_bimodal_eigen_proj ----
    def _bimodal_eigen_proj(day_df):
        from scipy.ndimage import gaussian_filter1d
        from scipy import signal
        prices = day_df['DealPrice'].values
        time_sec = day_df['DealTimeSecond'].values
        if len(prices) < 100:
            return 0.0
        bin_size = 300
        t_min = time_sec.min()
        bins = (time_sec - t_min) // bin_size
        unique_bins = np.unique(bins)
        if len(unique_bins) < 3:
            return 0.0
        projections = []
        for b in unique_bins:
            mask = bins == b
            p_bin = prices[mask]
            if len(p_bin) < 20:
                continue
            try:
                hist, bin_edges = np.histogram(p_bin, bins=50)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
                peaks_idx = signal.argrelextrema(smoothed, np.greater, order=3)[0]
                if len(peaks_idx) >= 2:
                    peak_heights = smoothed[peaks_idx]
                    top2_idx = peaks_idx[np.argsort(peak_heights)[-2:]]
                    top2_idx = np.sort(top2_idx)
                    peak_high, peak_low = bin_centers[top2_idx[1]], bin_centers[top2_idx[0]]
                else:
                    m = np.mean(p_bin)
                    peak_high, peak_low = m, m
            except:
                m = np.mean(p_bin)
                peak_high, peak_low = m, m
            mid = (peak_high + peak_low) / 2
            above = p_bin >= mid
            states = np.where(above, 0, 1).astype(int)
            if len(states) < 5:
                continue
            tm = _build_transition_matrix(states, n_states=2, laplace_smooth=1.0)
            pi = _stationary_distribution(tm)
            dp = np.diff(p_bin)
            if len(dp) < 3:
                continue
            ddp = np.diff(dp)
            accel_high = np.mean(ddp[above[2:]]) if np.sum(above[2:]) > 0 else 0.0
            accel_low = np.mean(ddp[~above[2:]]) if np.sum(~above[2:]) > 0 else 0.0
            if not (np.isfinite(accel_high) and np.isfinite(accel_low)):
                continue
            proj = np.dot(pi, np.array([accel_high, accel_low]))
            projections.append(proj)
        if len(projections) == 0:
            return 0.0
        return np.mean(projections)
    day_agg["raw_nld_bimodal_eigen_proj"] = df.groupby(["StockId", "Date_int"]).apply(_bimodal_eigen_proj).reindex(day_agg.index).values

    # ---- 3. f_nld_critical_slowing_phase_accel ----
    def _critical_slowing(day_df):
        prices = day_df['DealPrice'].values
        volumes = day_df['DealCount'].values.astype(float)
        prflag = day_df['PrFlag'].values if 'PrFlag' in day_df.columns else None
        if len(prices) < 100:
            return 0.0
        dp = np.diff(prices)
        bin_size = 100
        n_bins = len(dp) // bin_size
        if n_bins < 3:
            return 0.0
        phases = []
        for i in range(n_bins):
            s = i * bin_size
            e = s + bin_size
            dp_bin = dp[s:e]
            vol_bin = volumes[s+1:e+1]
            if len(dp_bin) < 5:
                continue
            try:
                ac1 = np.corrcoef(dp_bin[:-1], dp_bin[1:])[0, 1]
                if not np.isfinite(ac1):
                    ac1 = 0.0
            except:
                ac1 = 0.0
            ac1 = np.clip(ac1, -0.99, 0.99)
            vol_mean = np.mean(vol_bin) if len(vol_bin) > 0 else 1.0
            phi = np.arctan2(vol_mean, ac1)
            phases.append(phi)
        if len(phases) < 3:
            return 0.0
        phases = np.array(phases)
        d2phi = np.diff(phases, n=2)
        mean_d2phi = np.mean(d2phi)
        if not np.isfinite(mean_d2phi):
            return 0.0
        if prflag is not None:
            buy_vol = np.sum(volumes[prflag == 1])
            sell_vol = np.sum(volumes[prflag == 0])
            ofi_sign = np.sign(buy_vol - sell_vol)
        else:
            ofi_sign = np.sign(prices[-1] - prices[0])
        return mean_d2phi * ofi_sign
    day_agg["raw_nld_critical_slowing_phase_accel"] = df.groupby(["StockId", "Date_int"]).apply(_critical_slowing).reindex(day_agg.index).values

    # ---- 4. f_nld_dtb_singularity_cross ----
    def _dtb_singularity(day_df):
        prices = day_df['DealPrice'].values
        volumes = day_df['DealCount'].values.astype(float)
        if len(prices) < 30:
            return 0.0
        dp = np.diff(prices)
        dp_s = pd.Series(dp).ewm(span=5, min_periods=1).mean().values
        ddp = np.diff(dp_s)
        n = min(len(dp) - 1, len(ddp))
        if n < 10:
            return 0.0
        x = dp[1:n+1]
        y = dp_s[1:n+1]
        z = ddp[:n]
        # Use Z-component only (signed) to match original implementation
        cz = x[:-1]*y[1:] - y[:-1]*x[1:]
        cross_norm = cz
        dv = np.diff(volumes)
        dv_sign = np.sign(dv[2:n+1])
        min_len = min(len(cross_norm), len(dv_sign))
        if min_len == 0:
            return 0.0
        weighted = cross_norm[:min_len] * dv_sign[:min_len]
        return np.mean(weighted)
    day_agg["raw_nld_dtb_singularity_cross"] = df.groupby(["StockId", "Date_int"]).apply(_dtb_singularity).reindex(day_agg.index).values

    # ---- 5. f_nld_hopf_eigen_asymmetry ----
    def _hopf_eigen(day_df):
        prices = day_df['DealPrice'].values
        if len(prices) < 20:
            return 0.0
        dp = np.diff(prices)
        states = _price_to_three_states(dp)
        tm = _build_transition_matrix(states, n_states=3, laplace_smooth=1.0)
        max_ev = _leading_eigenvalue_real(tm)
        asym = tm[0, 0] - tm[2, 2]
        return max_ev * asym
    day_agg["raw_nld_hopf_eigen_asymmetry"] = df.groupby(["StockId", "Date_int"]).apply(_hopf_eigen).reindex(day_agg.index).values

    # ---- 6. f_nld_hysteresis_area ----
    def _hysteresis_area(day_df):
        prices = day_df['DealPrice'].values
        volumes = day_df['DealCount'].values.astype(float)
        if len(prices) < 50:
            return 0.0
        dp = np.diff(prices)
        dv = np.diff(volumes)
        dp_std = np.std(dp) + 1e-10
        dv_std = np.std(dv) + 1e-10
        dp_norm = dp / dp_std
        dv_norm = dv / dv_std
        bin_size = 50
        n_bins = len(dp_norm) // bin_size
        if n_bins < 2:
            return 0.0
        areas = []
        for i in range(n_bins):
            s = i * bin_size
            e = s + bin_size
            x = dp_norm[s:e]
            y = dv_norm[s:e]
            area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
            areas.append(area)
        return np.mean(areas)
    day_agg["raw_nld_hysteresis_area"] = df.groupby(["StockId", "Date_int"]).apply(_hysteresis_area).reindex(day_agg.index).values

    # ---- 7. f_nld_limit_cycle_collapse_markov ----
    def _limit_cycle_collapse(day_df):
        prices = day_df['DealPrice'].values
        if len(prices) < 20:
            return 0.0
        dp = np.diff(prices)
        states = _price_to_three_states(dp)
        tm = _build_transition_matrix(states, n_states=3, laplace_smooth=1.0)
        pi = _stationary_distribution(tm)
        projection = np.array([1.0, 0.0, -1.0])
        return np.dot(pi, projection)
    day_agg["raw_nld_limit_cycle_collapse_markov"] = df.groupby(["StockId", "Date_int"]).apply(_limit_cycle_collapse).reindex(day_agg.index).values

    # ---- 8. f_nld_limit_cycle_radius_accel ----
    def _limit_cycle_radius(day_df):
        prices = day_df['DealPrice'].values
        volumes = day_df['DealCount'].values.astype(float)
        if len(prices) < 20:
            return 0.0
        dp = np.diff(prices)
        dv = np.diff(volumes)
        dp_std = np.std(dp) + 1e-10
        dv_std = np.std(dv) + 1e-10
        dp_norm = dp / dp_std
        dv_norm = dv / dv_std
        R = np.sqrt(dp_norm**2 + dv_norm**2)
        R_series = pd.Series(R)
        R_ewm = R_series.ewm(span=10, min_periods=1).mean()
        d2R = R_ewm.diff().diff()
        mean_d2R = d2R.mean()
        if not np.isfinite(mean_d2R):
            return 0.0
        asym = np.sign(prices[-1] - prices[0])
        return mean_d2R * asym
    day_agg["raw_nld_limit_cycle_radius_accel"] = df.groupby(["StockId", "Date_int"]).apply(_limit_cycle_radius).reindex(day_agg.index).values

    # ---- 9. f_nld_lyapunov_divergence_accel ----
    def _lyapunov_divergence(day_df):
        prices = day_df['DealPrice'].values
        time_sec = day_df['DealTimeSecond'].values.astype(float)
        if len(prices) < 30:
            return 0.0
        p_std = np.std(prices) + 1e-10
        t_std = np.std(np.diff(time_sec)) + 1e-10
        p_norm = prices / p_std
        dt = np.diff(time_sec)
        dt_norm = dt / t_std
        k = 10
        if len(p_norm) < k + 20:
            return 0.0
        n = len(p_norm) - k
        D = np.zeros(n)
        for i in range(n):
            dp = p_norm[i + k] - p_norm[i]
            if i + k < len(dt_norm):
                dt_val = dt_norm[min(i + k - 1, len(dt_norm) - 1)]
            else:
                dt_val = 0
            D[i] = np.sqrt(dp**2 + dt_val**2)
        D_series = pd.Series(D)
        D_ewm = D_series.ewm(span=10, min_periods=1).mean()
        d2D = D_ewm.diff().diff()
        mean_d2D = d2D.mean()
        if not np.isfinite(mean_d2D):
            return 0.0
        asym = np.sign(prices[-1] - prices[0])
        return mean_d2D * asym
    day_agg["raw_nld_lyapunov_divergence_accel"] = df.groupby(["StockId", "Date_int"]).apply(_lyapunov_divergence).reindex(day_agg.index).values

    # ---- 10. f_nld_neimark_sacker_torus ----
    def _neimark_sacker(day_df):
        prices = day_df['DealPrice'].values
        volumes = day_df['DealCount'].values.astype(float)
        if len(prices) < 30:
            return 0.0
        cum_qty = np.cumsum(volumes)
        cum_pq = np.cumsum(prices * volumes)
        vwap = cum_pq / (cum_qty + 1e-10)
        dp = np.diff(prices)
        dv = np.diff(volumes)
        vwap_mid = vwap[1:]
        n = min(len(dp), len(dv), len(vwap_mid))
        if n < 10:
            return 0.0
        x = vwap_mid[:n] / (np.std(vwap_mid[:n]) + 1e-10)
        y = dp[:n] / (np.std(dp[:n]) + 1e-10)
        z = dv[:n] / (np.std(dv[:n]) + 1e-10)
        cx = y[:-1]*z[1:] - z[:-1]*y[1:]
        cy = z[:-1]*x[1:] - x[:-1]*z[1:]
        cz = x[:-1]*y[1:] - y[:-1]*x[1:]
        norm = np.sqrt(cx**2 + cy**2 + cz**2)
        if len(norm) < 3:
            return 0.0
        d2_norm = np.diff(norm, n=2)
        mean_d2 = np.mean(d2_norm)
        if not np.isfinite(mean_d2):
            return 0.0
        asym = np.sign(prices[-1] - prices[0])
        return mean_d2 * asym
    day_agg["raw_nld_neimark_sacker_torus"] = df.groupby(["StockId", "Date_int"]).apply(_neimark_sacker).reindex(day_agg.index).values

    # ---- 11. f_nld_saddle_loop_divergence ----
    def _saddle_loop(day_df):
        prices = day_df['DealPrice'].values
        volumes = day_df['DealCount'].values.astype(float)
        if len(prices) < 50:
            return 0.0
        cum_qty = np.cumsum(volumes)
        cum_pq = np.cumsum(prices * volumes)
        vwap = cum_pq / (cum_qty + 1e-10)
        diff_sign = np.sign(prices - vwap)
        crossings = np.where(np.diff(diff_sign) != 0)[0]
        if len(crossings) < 4:
            return 0.0
        intervals = np.diff(crossings).astype(float)
        if len(intervals) < 3:
            return 0.0
        d2_intervals = np.diff(intervals, n=2)
        if len(d2_intervals) == 0:
            return 0.0
        last_dev = prices[-1] - vwap[-1]
        mean_d2 = np.mean(d2_intervals)
        if not np.isfinite(mean_d2):
            return 0.0
        return mean_d2 * np.sign(last_dev)
    day_agg["raw_nld_saddle_loop_divergence"] = df.groupby(["StockId", "Date_int"]).apply(_saddle_loop).reindex(day_agg.index).values

    # ---- 12. f_nld_saddle_node_phase_accel ----
    def _saddle_node_phase(day_df):
        if 'BuyPr' not in day_df.columns or 'SellPr' not in day_df.columns:
            return np.nan
        prices = day_df['DealPrice'].values
        buy_pr = day_df['BuyPr'].values
        sell_pr = day_df['SellPr'].values
        if len(prices) < 20:
            return 0.0
        x = prices - buy_pr
        y = sell_pr - prices
        x = np.maximum(x, 1e-4)
        y = np.maximum(y, 1e-4)
        phi = np.arctan2(y, x)
        phi_series = pd.Series(phi)
        phi_ewm = phi_series.ewm(span=10, min_periods=1).mean()
        d2phi = phi_ewm.diff().diff()
        d2phi_clean = d2phi.dropna()
        if len(d2phi_clean) == 0:
            return 0.0
        abs_max_idx = d2phi_clean.abs().idxmax()
        return d2phi_clean.loc[abs_max_idx]
    day_agg["raw_nld_saddle_node_phase_accel"] = df.groupby(["StockId", "Date_int"]).apply(_saddle_node_phase).reindex(day_agg.index).values

    # ---- 13. f_nld_stochastic_bifurcation_phase ----
    def _stochastic_bifurcation(day_df):
        from scipy.ndimage import gaussian_filter1d
        from scipy import signal
        prices = day_df['DealPrice'].values
        if len(prices) < 50:
            return 0.0
        try:
            hist, bin_edges = np.histogram(prices, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
            peaks_idx = signal.argrelextrema(smoothed, np.greater, order=3)[0]
            if len(peaks_idx) >= 2:
                peak_heights = smoothed[peaks_idx]
                top2_idx = peaks_idx[np.argsort(peak_heights)[-2:]]
                top2_idx = np.sort(top2_idx)
                peak_high, peak_low = bin_centers[top2_idx[1]], bin_centers[top2_idx[0]]
            else:
                m = np.mean(prices)
                peak_high, peak_low = m, m
        except:
            m = np.mean(prices)
            peak_high, peak_low = m, m
        mode_vec = np.array([peak_high, peak_low])
        qty = day_df['DealCount'].values.astype(float)
        cum_qty = np.cumsum(qty)
        cum_pq = np.cumsum(prices * qty)
        vwap = cum_pq / (cum_qty + 1e-10)
        mid = len(vwap) // 2
        if mid < 5:
            return 0.0
        vwap_series = pd.Series(vwap)
        vwap_ewm = vwap_series.ewm(span=20, min_periods=1).mean()
        d2_vwap = vwap_ewm.diff().diff()
        first_half_accel = d2_vwap.iloc[:mid].mean()
        second_half_accel = d2_vwap.iloc[mid:].mean()
        if not (np.isfinite(first_half_accel) and np.isfinite(second_half_accel)):
            return 0.0
        accel_vec = np.array([first_half_accel, second_half_accel])
        return np.dot(mode_vec, accel_vec)
    day_agg["raw_nld_stochastic_bifurcation_phase"] = df.groupby(["StockId", "Date_int"]).apply(_stochastic_bifurcation).reindex(day_agg.index).values

    # ---- 14. f_nld_stochastic_focusing_phase ----
    def _stochastic_focusing(day_df):
        prices = day_df['DealPrice'].values
        volumes = day_df['DealCount'].values.astype(float)
        if len(prices) < 30:
            return 0.0
        dp = np.diff(prices)
        dv = np.diff(volumes)
        dp_std = np.std(dp) + 1e-10
        dv_std = np.std(dv) + 1e-10
        theta = np.arctan2(dp / dp_std, dv / dv_std)
        theta_series = pd.Series(theta)
        theta_ewm = theta_series.ewm(span=10, min_periods=1).mean()
        d2theta = theta_ewm.diff().diff()
        mean_d2 = d2theta.mean()
        if not np.isfinite(mean_d2):
            return 0.0
        asym = np.sign(prices[-1] - prices[0])
        return mean_d2 * asym
    day_agg["raw_nld_stochastic_focusing_phase"] = df.groupby(["StockId", "Date_int"]).apply(_stochastic_focusing).reindex(day_agg.index).values

    # ---- 15. f_nld_transient_attractor_proj ----
    def _transient_attractor(day_df):
        if 'CeilPr' not in day_df.columns or 'FloorPr' not in day_df.columns:
            return np.nan
        prices = day_df['DealPrice'].values
        ceil_pr = day_df['CeilPr'].values
        floor_pr = day_df['FloorPr'].values
        if len(prices) < 20:
            return 0.0
        price_series = pd.Series(prices)
        price_ewm = price_series.ewm(span=10, min_periods=1).mean()
        d2p = price_ewm.diff().diff()
        d2p_vals = d2p.values
        dist_up = ceil_pr - prices
        dist_dn = prices - floor_pr
        attractor_axis = dist_up - dist_dn
        scale = (ceil_pr[0] - floor_pr[0]) if (ceil_pr[0] - floor_pr[0]) > 0 else 1.0
        attractor_norm = attractor_axis / scale
        n = min(len(d2p_vals), len(attractor_norm))
        valid_mask = np.isfinite(d2p_vals[:n])
        if np.sum(valid_mask) == 0:
            return 0.0
        proj = d2p_vals[:n][valid_mask] * attractor_norm[:n][valid_mask]
        return np.mean(proj)
    day_agg["raw_nld_transient_attractor_proj"] = df.groupby(["StockId", "Date_int"]).apply(_transient_attractor).reindex(day_agg.index).values

    # === End Alpha NLD: 16 New Nonlinear Dynamics Features ===

    res = day_agg.reset_index()
    res.rename(columns={"Date_int": "Date"}, inplace=True)
    res["StockId"] = res["StockId"].astype(str)
    
    return res

if __name__ == "__main__":
    import os
    DATA_DIR = r"G:\AI_dataAnalysis\data"
    f_tick = os.path.join(DATA_DIR, "trade_level1_data", "1101.parquet")
    df_tick = pd.read_parquet(f_tick, filters=[("Date", ">=", "20140101"), ("Date", "<=", "20251231")])
    df_tick["Date"] = df_tick["Date"].astype(int)
    
    print("Running new logic...")
    t0 = time.time()
    out = preprocess_single_stock_tick_opt(df_tick)
    t1 = time.time()
    print(f"Vectorized took: {t1 - t0:.4f} seconds.")
    
    out.to_parquet("G:/module_rewrite/test_data/baseline_1101_tick_opt_NEW.parquet")
    print("Done")
