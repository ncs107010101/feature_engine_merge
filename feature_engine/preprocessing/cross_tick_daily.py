"""
feature_engine.preprocessing.cross_tick_daily
==========================================
Preprocessing for features requiring both tick data and daily OHLC data.

This module:
1. Reuses preprocess_single_stock_tick to get all tick intermediates
2. Merges daily OHLC columns from daily_data kwargs
3. Computes shared cross intermediates needed by all features

Shared intermediates precomputed:
- raw_consecutive_up_days: Consecutive up-days from daily close vs prev close
- raw_consecutive_up_vol: Average daily volume over consecutive up days
- raw_mode_share: Volume share at mode (most traded) price
- raw_mode_price: Mode (most traded) price
- raw_patch_thickness: Mode ±0.5% volume share
- raw_push_speed: (VWAP - Open) / Open
- raw_cross_count: VWAP crossing count
- raw_cross_vol: Volume of ticks crossing VWAP
- raw_cross_both_vol: Volume of active buy ticks crossing both VWAP and open
- raw_total_active_buy: Total active buy volume
- raw_active_sell: Total active sell volume
- raw_first500_sell_ratio: Sell ratio in first 500 ticks per day
- raw_prior_pessimism: Binary gap-down flag (open < prev_close * 0.99)
- raw_ignorant_selling: Sum of small_sell in rebound bins (BIN_SIZE=50)
- raw_total_vol_50bin: Total vol for normalization (BIN_SIZE=50)
- raw_neg_mood: Binary severe-drop flag (open < prev_close * 0.97)
- raw_mood_sell: Sum of small_sell in up bins (BIN_SIZE=100)
- raw_gap_down: Binary gap-down flag (open < prev_close * 0.98)
- raw_instant_exit: Sum of small_sell in first 5 bins (BIN_SIZE=100)
- raw_loss_domain_max: Max of (loss_domain × risk_seeking) per day (BIN_SIZE=100)
- raw_high_prior: Binary (20d return > 15%)
- raw_bin1_large_buy: Sum of large_buy in first bin (BIN_SIZE=100)
- raw_small_buy_ticks: DealCount==1 & PrFlag==1 daily count (F39)
- raw_first500_small_buy: First 500 ticks, PrFlag==1, DealCount<=Small_Thresh (F42)
- raw_last500_small_sell: Last 500 ticks, PrFlag==0, DealCount<=Small_Thresh (F41)
- raw_bin200_garbling_dumping: 200-tick bin garbling × dumping (F43)
- raw_salience_trap_max: Daily max of (salience/vol_quality) (F40)
- raw_loss_domain_vol_max: Daily max of (loss_domain×volatility×sell_vol) (F45)
- raw_bin50_bin1_large_buy: 50-tick bin 1 large buy (F44)
- raw_bin50_total_large_buy: 50-tick total large buy (F44)
- raw_vol20: 20-day return std (F41)
- raw_vol20_q20: 20-day vol 20th percentile (F41)
- raw_high_shift1: Previous day's high (F41)
- raw_ret1: Daily return
- raw_ret1_shift1: Previous day's return (F42)
- raw_gap_down: Binary gap-down (F42)
- raw_ret5: 5-day return (F43)
- raw_screening_premium: (open_p - prev_high) / prev_high (F44)
"""

import pandas as pd
import numpy as np
from .single_stock_tick import preprocess_single_stock_tick


def preprocess_cross_tick_daily(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Preprocessing for cross_tick_daily features.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw tick data with columns: StockId, Date, HHMMSS, DealTimeSecond, 
        DealPrice, DealCount, TotalQty, PrFlag, etc.
    **kwargs
        - daily_data: DataFrame with daily OHLC data
        
    Returns
    -------
    pd.DataFrame
        Merged tick+daily intermediates keyed by (StockId, Date)
    """
    if "daily_data" not in kwargs:
        raise ValueError(
            "cross_tick_daily requires daily OHLC data passed as "
            "`daily_data` in kwargs."
        )

    df_daily = kwargs["daily_data"].copy()
    
    # Validate required daily columns
    req_daily = ["StockId", "Date", "收盤價", "開盤價", "最高價"]
    missing = [c for c in req_daily if c not in df_daily.columns]
    if missing:
        raise ValueError(
            f"daily_data is missing required columns: {missing}. "
            "cross_tick_daily requires '收盤價' and '開盤價'."
        )
    
    tick_intermediates = preprocess_single_stock_tick(data)
    tick_intermediates["raw_active_sell"] = tick_intermediates["sell_vol"]
    tick_intermediates["raw_total_active_buy"] = tick_intermediates["buy_vol"]

    df_daily["Date"] = df_daily["Date"].astype(int)
    tick_intermediates["Date"] = tick_intermediates["Date"].astype(int)
    
    # --- Step 3: Merge daily OHLC ---
    merge_cols = ["StockId", "Date"]
    daily_cols_needed_base = ["收盤價", "開盤價", "成交量(千股)", "報酬率", "最高價"]
    daily_cols_needed_optional = ["融券餘額(張)"]
    
    # Filter to only include columns that exist
    daily_cols_available = [c for c in daily_cols_needed_base + daily_cols_needed_optional if c in df_daily.columns]
    daily_cols_needed = daily_cols_available
    
    # Compute 報酬率 if missing (derived from intra-day return: (收盤價 - 開盤價) / 開盤價)
    if "報酬率" not in df_daily.columns:
        df_daily["報酬率"] = (df_daily["收盤價"] - df_daily["開盤價"]) / (df_daily["開盤價"] + 1e-9) * 100
    
    tick_intermediates = tick_intermediates.merge(
        df_daily[merge_cols + daily_cols_needed],
        on=merge_cols,
        how="left",
    )
    
    df = tick_intermediates.sort_values(["StockId", "Date"]).reset_index(drop=True)
    
    df = _compute_consecutive_up_days(df)
    df = _compute_mode_stats(data, df)
    df["raw_push_speed"] = np.where(
        df["開盤價"] > 0,
        (df["vwap"] - df["開盤價"]) / df["開盤價"],
        0.0
    )
    df = _compute_tick_crossings(data, df)
    df = _compute_first500_sell_ratio(data, df)
    
    # 新增: F02, F05, F09, F10, F14 intermediates
    df = _compute_bin_aggs_50(data, df)
    df = _compute_bin_aggs_100(data, df)
    
    # 新增: F39-F45 intermediates (F40-F45 use tick+daily, F39 uses broker+tick+daily)
    df = _compute_small_buy_ticks(data, df)  # F39: DealCount==1 & PrFlag==1
    df = _compute_first500_small_buy(data, df)  # F42: first 500 ticks small active buy
    df = _compute_last500_small_sell(data, df)  # F41: last 500 ticks small active sell
    df = _compute_bin_aggs_200(data, df)  # F43: 200-tick bins
    df = _compute_bin_aggs_50_extended(data, df)  # F40, F44, F45: extended 50-tick bin features
    df = _compute_vol_surface(df)  # F41: vol20, vol20_q20, high_shift1
    df = _compute_ret_daily(df)  # F42, F43: ret1_shift1, gap_down, ret5
    df = _compute_screening_premium(df)  # F44: screening_premium
    
    # 新增: F41-F50 AFD intermediates (冷區、融券槓桿等)
    df = _compute_afd_intermediates(data, df)
    df = _compute_afd_tick_level_intermediates(data, df)
    
    # 新增: 融券餘額(張) from daily data
    df = _merge_margin_balance(df_daily, df)
    
    return df


def _compute_tick_crossings(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    tick_data["_prev_price"] = tick_data.groupby(["StockId", "Date_int"])["DealPrice"].shift(1).fillna(0)
    tick_data["_prev_price"] = np.where(
        tick_data["_prev_price"] == 0, tick_data["DealPrice"], tick_data["_prev_price"]
    )
    tick_data["_vwap_key"] = tick_data.set_index(["StockId", "Date_int"]).index.map(
        df.set_index(["StockId", "Date"])["vwap"]
    )
    tick_data["_open_key"] = tick_data.set_index(["StockId", "Date_int"]).index.map(
        df.set_index(["StockId", "Date"])["開盤價"]
    )
    tick_data["_max_thresh"] = np.maximum(tick_data["_vwap_key"], tick_data["_open_key"])
    tick_data["_min_thresh"] = np.minimum(tick_data["_vwap_key"], tick_data["_open_key"])
    tick_data["_above_vwap"] = (tick_data["DealPrice"] >= tick_data["_vwap_key"]).astype(int)
    tick_data["_prev_above_vwap"] = tick_data.groupby(["StockId", "Date_int"])["_above_vwap"].shift(1).fillna(0).astype(int)
    tick_data["_vwap_cross"] = (
        (tick_data["_above_vwap"] != tick_data["_prev_above_vwap"]) &
        (tick_data["_prev_above_vwap"].notna())
    ).astype(int)
    tick_data["_cross_both"] = (
        (tick_data["PrFlag"] == 1) &
        (tick_data["_prev_price"] < tick_data["_min_thresh"]) &
        (tick_data["DealPrice"] > tick_data["_max_thresh"])
    ).astype(int)
    cross_count = tick_data.groupby(["StockId", "Date_int"])["_vwap_cross"].sum().reset_index()
    cross_count.columns = ["StockId", "Date_int", "raw_cross_count"]
    cross_vol = tick_data[tick_data["_vwap_cross"] == 1].groupby(["StockId", "Date_int"])["DealCount"].sum().reset_index()
    cross_vol.columns = ["StockId", "Date_int", "raw_cross_vol"]
    cross_both_vol = tick_data[tick_data["_cross_both"] == 1].groupby(["StockId", "Date_int"])["DealCount"].sum().reset_index()
    cross_both_vol.columns = ["StockId", "Date_int", "raw_cross_both_vol"]
    cross_stats = cross_count.merge(cross_vol, on=["StockId", "Date_int"], how="outer")
    cross_stats = cross_stats.merge(cross_both_vol, on=["StockId", "Date_int"], how="outer")
    cross_stats = cross_stats.rename(columns={"Date_int": "Date"})
    df = df.merge(
        cross_stats[["StockId", "Date", "raw_cross_count", "raw_cross_vol", "raw_cross_both_vol"]],
        on=["StockId", "Date"],
        how="left"
    )
    df["raw_cross_count"] = df["raw_cross_count"].fillna(0)
    df["raw_cross_vol"] = df["raw_cross_vol"].fillna(0)
    df["raw_cross_both_vol"] = df["raw_cross_both_vol"].fillna(0)
    return df


def _compute_first500_sell_ratio(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    first500_ticks = tick_data.groupby(["StockId", "Date_int"]).head(500)
    pivot_table = first500_ticks.groupby(["StockId", "Date_int", "PrFlag"])["DealCount"].sum().reset_index()
    pivot_table = pivot_table.pivot(
        index=["StockId", "Date_int"],
        columns="PrFlag",
        values="DealCount"
    ).fillna(0)
    pivot_table["total_volume"] = pivot_table.sum(axis=1)
    pivot_table["raw_first500_sell_ratio"] = pivot_table[0] / (pivot_table["total_volume"] + 1e-8)
    pivot_table = pivot_table.reset_index()
    pivot_table = pivot_table.rename(columns={"Date_int": "Date"})
    df = df.merge(
        pivot_table[["StockId", "Date", "raw_first500_sell_ratio"]],
        on=["StockId", "Date"],
        how="left"
    )
    df["raw_first500_sell_ratio"] = df["raw_first500_sell_ratio"].fillna(0)
    return df


def _compute_tick_crossings(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    tick_data["_prev_price"] = tick_data.groupby(["StockId", "Date_int"])["DealPrice"].shift(1).fillna(0)
    tick_data["_prev_price"] = np.where(
        tick_data["_prev_price"] == 0, tick_data["DealPrice"], tick_data["_prev_price"]
    )
    tick_data["_vwap_key"] = tick_data.set_index(["StockId", "Date_int"]).index.map(
        df.set_index(["StockId", "Date"])["vwap"]
    )
    tick_data["_open_key"] = tick_data.set_index(["StockId", "Date_int"]).index.map(
        df.set_index(["StockId", "Date"])["開盤價"]
    )
    tick_data["_max_thresh"] = np.maximum(tick_data["_vwap_key"], tick_data["_open_key"])
    tick_data["_min_thresh"] = np.minimum(tick_data["_vwap_key"], tick_data["_open_key"])
    tick_data["_above_vwap"] = (tick_data["DealPrice"] >= tick_data["_vwap_key"]).astype(int)
    tick_data["_prev_above_vwap"] = tick_data.groupby(["StockId", "Date_int"])["_above_vwap"].shift(1).fillna(0).astype(int)
    tick_data["_vwap_cross"] = (
        (tick_data["_above_vwap"] != tick_data["_prev_above_vwap"]) &
        (tick_data["_prev_above_vwap"].notna())
    ).astype(int)
    tick_data["_cross_both"] = (
        (tick_data["PrFlag"] == 1) &
        (tick_data["_prev_price"] < tick_data["_min_thresh"]) &
        (tick_data["DealPrice"] > tick_data["_max_thresh"])
    ).astype(int)
    cross_count = tick_data.groupby(["StockId", "Date_int"])["_vwap_cross"].sum().reset_index()
    cross_count.columns = ["StockId", "Date_int", "raw_cross_count"]
    cross_vol = tick_data[tick_data["_vwap_cross"] == 1].groupby(["StockId", "Date_int"])["DealCount"].sum().reset_index()
    cross_vol.columns = ["StockId", "Date_int", "raw_cross_vol"]
    cross_both_vol = tick_data[tick_data["_cross_both"] == 1].groupby(["StockId", "Date_int"])["DealCount"].sum().reset_index()
    cross_both_vol.columns = ["StockId", "Date_int", "raw_cross_both_vol"]
    cross_stats = cross_count.merge(cross_vol, on=["StockId", "Date_int"], how="outer")
    cross_stats = cross_stats.merge(cross_both_vol, on=["StockId", "Date_int"], how="outer")
    cross_stats = cross_stats.rename(columns={"Date_int": "Date"})
    df = df.merge(
        cross_stats[["StockId", "Date", "raw_cross_count", "raw_cross_vol", "raw_cross_both_vol"]],
        on=["StockId", "Date"],
        how="left"
    )
    df["raw_cross_count"] = df["raw_cross_count"].fillna(0)
    df["raw_cross_vol"] = df["raw_cross_vol"].fillna(0)
    df["raw_cross_both_vol"] = df["raw_cross_both_vol"].fillna(0)
    return df


def _compute_consecutive_up_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev_close = df.groupby("StockId")["收盤價"].shift(1)
    df["_is_up"] = (df["收盤價"] > prev_close).fillna(0).astype(int)
    df["_up_change"] = (df["_is_up"] != df["_is_up"].shift(1)).cumsum()
    df["raw_consecutive_up_days"] = 0
    mask = df["_is_up"] == 1
    df.loc[mask, "raw_consecutive_up_days"] = (
        df[mask].groupby(["StockId", "_up_change"]).cumcount() + 1
    ).values
    daily_vol_col = "成交量(千股)"
    if daily_vol_col in df.columns:
        df["_daily_vol"] = df[daily_vol_col] * 1000
    else:
        df["_daily_vol"] = 0.0
    daily_data = df.groupby(["StockId", "Date"]).agg(
        daily_vol=("_daily_vol", "first"),
        consecutive_up_days=("raw_consecutive_up_days", "first"),
    ).reset_index()
    def calc_up_avg_vol(group):
        n = len(group)
        result = np.zeros(n)
        for i in range(n):
            days = int(group.iloc[i]["consecutive_up_days"])
            if days > 0:
                start_idx = max(0, i - days + 1)
                result[i] = group.iloc[start_idx:i+1]["daily_vol"].mean()
        return pd.Series(result, index=group.index)
    raw = daily_data.groupby("StockId").apply(lambda x: calc_up_avg_vol(x))
    if isinstance(raw, pd.DataFrame):
        # Single-stock case: (1, n_dates) → transpose to (n_dates,) Series
        raw = raw.T.iloc[:, 0]
    elif isinstance(raw.index, pd.MultiIndex):
        # Multi-stock case: drop outer StockId level
        raw = raw.droplevel(0)
    daily_data["raw_consecutive_up_vol"] = raw
    df = df.merge(
        daily_data[["StockId", "Date", "raw_consecutive_up_vol"]],
        on=["StockId", "Date"],
        how="left"
    )
    df["raw_consecutive_up_vol"] = df["raw_consecutive_up_vol"].fillna(0)
    df.drop(columns=["_is_up", "_up_change", "_daily_vol"], inplace=True, errors="ignore")
    return df


def _compute_mode_stats(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    pv_agg = tick_data.groupby(["StockId", "Date_int", "DealPrice"]).agg(
        price_vol=("DealCount", "sum")
    ).reset_index()
    total_vol = pv_agg.groupby(["StockId", "Date_int"])["price_vol"].sum().reset_index()
    total_vol.columns = ["StockId", "Date_int", "total_price_vol"]
    pv_agg = pv_agg.merge(total_vol, on=["StockId", "Date_int"])
    idx_max = pv_agg.groupby(["StockId", "Date_int"])["price_vol"].idxmax()
    mode_stats = pv_agg.loc[idx_max, ["StockId", "Date_int", "DealPrice", "price_vol"]].copy()
    mode_stats = mode_stats.rename(columns={
        "DealPrice": "raw_mode_price",
        "price_vol": "raw_mode_vol"
    })
    _tv = total_vol.copy()
    _tv.columns = ["StockId", "Date_int", "total_price_vol_tv"]
    mode_stats = mode_stats.merge(_tv, on=["StockId", "Date_int"])
    mode_stats["raw_mode_share"] = mode_stats["raw_mode_vol"] / (mode_stats["total_price_vol_tv"] + 1e-8)
    pv_agg = pv_agg.merge(mode_stats[["StockId", "Date_int", "raw_mode_price"]], on=["StockId", "Date_int"])
    lower = pv_agg["DealPrice"] >= pv_agg["raw_mode_price"] * 0.995
    upper = pv_agg["DealPrice"] <= pv_agg["raw_mode_price"] * 1.005
    pv_agg["_in_patch"] = (lower & upper).astype(int)
    pv_agg["_patch_vol"] = pv_agg["price_vol"] * pv_agg["_in_patch"]
    patch_vol = pv_agg.groupby(["StockId", "Date_int"])["_patch_vol"].sum().reset_index()
    patch_vol.columns = ["StockId", "Date_int", "raw_patch_vol"]
    mode_stats = mode_stats.merge(patch_vol, on=["StockId", "Date_int"])
    mode_stats["raw_patch_thickness"] = mode_stats["raw_patch_vol"] / (mode_stats["total_price_vol_tv"] + 1e-8)
    mode_stats = mode_stats.rename(columns={"Date_int": "Date"})
    df = df.merge(
        mode_stats[["StockId", "Date", "raw_mode_share", "raw_mode_price", "raw_patch_thickness"]],
        on=["StockId", "Date"],
        how="left"
    )
    df["raw_mode_share"] = df["raw_mode_share"].fillna(0)
    df["raw_mode_price"] = df["raw_mode_price"].fillna(df["vwap"])
    df["raw_patch_thickness"] = df["raw_patch_thickness"].fillna(0)
    return df


def _make_bins_vectorized(df: pd.DataFrame, bin_size: int) -> pd.Series:
    """
    向量化版本的 bin 標籤生成。
    假設 df 已按 (Date, TotalQty) 排序。
    返回 Series，值為日內 bin 編號，不足 bin_size 的為 -1。
    """
    # 日內順序編號
    cumcount = df.groupby('Date_int').cumcount()
    # 日內總筆數
    day_sizes = df.groupby('Date_int')['Date_int'].transform('size')
    n_full = (day_sizes // bin_size) * bin_size
    # 有效 tick (在完整 bin 範圍內)
    valid_mask = cumcount < n_full
    bins = pd.Series(-1, index=df.index, dtype=int)
    bins[valid_mask] = cumcount[valid_mask] // bin_size
    return bins


def _compute_bin_aggs_50(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    計算 BIN_SIZE=50 的 bin 等級聚合，用於 F02 (f_be_prior_bias_neglect)。
    產出 intermediates:
    - raw_prior_pessimism: open < prev_close * 0.99
    - raw_ignorant_selling: 小單賣出在反彈區間的總和
    - raw_total_vol_50bin: 總成交量用於正規化
    """
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    # 獲取 Small_Thresh
    if "Small_Thresh" not in tick_data.columns and "Small_Thresh" in df.columns:
        thresh_df = df[["StockId", "Date", "Small_Thresh"]].copy()
        thresh_df["Date_int"] = thresh_df["Date"].astype(int)
        tick_data = tick_data.merge(
            thresh_df[["StockId", "Date_int", "Small_Thresh"]],
            on=["StockId", "Date_int"],
            how="left"
        )
        tick_data["Small_Thresh"] = tick_data["Small_Thresh"].fillna(1)
    
    # 計算 tick 等級 flags
    tick_data["_pv"] = tick_data["DealPrice"] * tick_data["DealCount"]
    tick_data["_is_buy"] = (tick_data["PrFlag"] == 1).astype(int)
    tick_data["_is_sell"] = (tick_data["PrFlag"] == 0).astype(int)
    tick_data["_is_buy_vol"] = tick_data["DealCount"] * tick_data["_is_buy"]
    tick_data["_is_sell_vol"] = tick_data["DealCount"] * tick_data["_is_sell"]
    
    # Small_Thresh 閾值
    small_thresh = tick_data.get("Small_Thresh")
    if small_thresh is not None:
        small_thresh_val = small_thresh.values
    else:
        small_thresh_val = 1
    tick_data["_small_sell"] = (
        (tick_data["_is_sell"] == 1) & 
        (tick_data["DealCount"].values <= small_thresh_val)
    ).astype(int) * tick_data["DealCount"]
    
    # 生成 bin 標籤 (BIN_SIZE=50)
    tick_data["_bin_50"] = _make_bins_vectorized(tick_data, 50)
    valid = tick_data[tick_data["_bin_50"] >= 0].copy()
    
    if valid.empty:
        return df
    
    # Per-bin aggregation
    bin_agg = valid.groupby(["StockId", "Date_int", "_bin_50"]).agg(
        bin_open=("DealPrice", "first"),
        bin_close=("DealPrice", "last"),
        small_sell=("_small_sell", "sum"),
        total_vol=("DealCount", "sum"),
    ).reset_index()
    
    # 反彈區間：收盤 > 開盤
    bin_agg["rebound"] = (bin_agg["bin_close"] > bin_agg["bin_open"]).astype(int)
    
    # 計算每日總成交量
    daily_total = valid.groupby(["StockId", "Date_int"])["DealCount"].sum().reset_index()
    daily_total.columns = ["StockId", "Date_int", "raw_total_vol_50bin"]
    
    # 計算反彈區間的小單賣出總和
    rebound_bins = bin_agg[bin_agg["rebound"] == 1]
    ignorant_selling = rebound_bins.groupby(["StockId", "Date_int"])["small_sell"].sum().reset_index()
    ignorant_selling.columns = ["StockId", "Date_int", "raw_ignorant_selling"]
    
    # 計算前一日收盤價
    df_sorted = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
    df_sorted["_prev_close"] = df_sorted.groupby("StockId")["收盤價"].shift(1)
    
    # prior_pessimism: open < prev_close * 0.99
    df_sorted["raw_prior_pessimism"] = (
        (df_sorted["開盤價"] < df_sorted["_prev_close"] * 0.99)
    ).fillna(0).astype(int)
    
    # 合併到 df
    prior_pessimism = df_sorted[["StockId", "Date", "raw_prior_pessimism"]].copy()
    ignorant_selling = ignorant_selling.rename(columns={"Date_int": "Date"})
    daily_total = daily_total.rename(columns={"Date_int": "Date"})
    df = df.merge(prior_pessimism, on=["StockId", "Date"], how="left")
    df = df.merge(ignorant_selling[["StockId", "Date", "raw_ignorant_selling"]], on=["StockId", "Date"], how="left")
    df = df.merge(daily_total[["StockId", "Date", "raw_total_vol_50bin"]], on=["StockId", "Date"], how="left")
    df["raw_prior_pessimism"] = df["raw_prior_pessimism"].fillna(0)
    df["raw_ignorant_selling"] = df["raw_ignorant_selling"].fillna(0)
    df["raw_total_vol_50bin"] = df["raw_total_vol_50bin"].fillna(0)
    
    return df


def _compute_bin_aggs_100(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    計算 BIN_SIZE=100 的 bin 等級聚合，用於 F05, F09, F10, F14。
    產出 intermediates:
    - raw_neg_mood: open < prev_close * 0.97 (for F05)
    - raw_mood_sell: 小單賣出在上漲區間的總和 (for F05)
    - raw_gap_down: open < prev_close * 0.98 (for F09)
    - raw_instant_exit: 前5個bin的小單賣出總和 (for F09)
    - raw_loss_domain_max: (VWAP < open * 0.98) × (std/mean) 的日最大值 (for F10)
    - raw_high_prior: close_now/close_20 - 1 > 0.15 (for F14)
    - raw_bin1_large_buy: 第一個bin的大單買入總和 (for F14)
    """
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    # 獲取 Small_Thresh 和 Large_Thresh
    if "Small_Thresh" not in tick_data.columns and "Small_Thresh" in df.columns:
        thresh_df = df[["StockId", "Date", "Small_Thresh", "Large_Thresh"]].copy()
        thresh_df["Date_int"] = thresh_df["Date"].astype(int)
        tick_data = tick_data.merge(
            thresh_df[["StockId", "Date_int", "Small_Thresh", "Large_Thresh"]],
            on=["StockId", "Date_int"],
            how="left"
        )
        tick_data["Small_Thresh"] = tick_data["Small_Thresh"].fillna(1)
        tick_data["Large_Thresh"] = tick_data["Large_Thresh"].fillna(1)
    
    # 計算 tick 等級 flags
    tick_data["_pv"] = tick_data["DealPrice"] * tick_data["DealCount"]
    tick_data["_is_buy"] = (tick_data["PrFlag"] == 1).astype(int)
    tick_data["_is_sell"] = (tick_data["PrFlag"] == 0).astype(int)
    tick_data["_is_buy_vol"] = tick_data["DealCount"] * tick_data["_is_buy"]
    tick_data["_is_sell_vol"] = tick_data["DealCount"] * tick_data["_is_sell"]
    
    # Small/Large 閾值
    small_thresh = tick_data.get("Small_Thresh")
    large_thresh = tick_data.get("Large_Thresh")
    if small_thresh is not None:
        small_thresh_val = small_thresh.values
    else:
        small_thresh_val = 1
    if large_thresh is not None:
        large_thresh_val = large_thresh.values
    else:
        large_thresh_val = 1
    tick_data["_small_sell"] = (
        (tick_data["_is_sell"] == 1) & 
        (tick_data["DealCount"].values <= small_thresh_val)
    ).astype(int) * tick_data["DealCount"]
    tick_data["_large_buy"] = (
        (tick_data["_is_buy"] == 1) & 
        (tick_data["DealCount"].values >= large_thresh_val)
    ).astype(int) * tick_data["DealCount"]
    
    # 生成 bin 標籤 (BIN_SIZE=100)
    tick_data["_bin_100"] = _make_bins_vectorized(tick_data, 100)
    valid = tick_data[tick_data["_bin_100"] >= 0].copy()
    
    if valid.empty:
        return df
    
    # Per-bin aggregation
    bin_agg = valid.groupby(["StockId", "Date_int", "_bin_100"]).agg(
        bin_open=("DealPrice", "first"),
        bin_close=("DealPrice", "last"),
        bin_vwap_num=("_pv", "sum"),
        bin_vwap_den=("DealCount", "sum"),
        small_sell=("_small_sell", "sum"),
        large_buy=("_large_buy", "sum"),
    ).reset_index()
    
    # Bin VWAP
    bin_agg["bin_vwap"] = bin_agg["bin_vwap_num"] / (bin_agg["bin_vwap_den"] + 1e-8)
    
    # 上漲區間：收盤 > 開盤
    bin_agg["up"] = (bin_agg["bin_close"] > bin_agg["bin_open"]).astype(int)
    
    # 計算 mood_sell (上漲區間的小單賣出)
    up_bins = bin_agg[bin_agg["up"] == 1]
    mood_sell = up_bins.groupby(["StockId", "Date_int"])["small_sell"].sum().reset_index()
    mood_sell.columns = ["StockId", "Date_int", "raw_mood_sell"]
    
    # 計算 instant_exit (前5個bin的小單賣出)
    first5 = valid[valid["_bin_100"] < 5]
    instant_exit = first5.groupby(["StockId", "Date_int"])["_small_sell"].sum().reset_index()
    instant_exit.columns = ["StockId", "Date_int", "raw_instant_exit"]
    
    # 計算 bin1_large_buy (第一個bin的大單買入)
    bin1 = valid[valid["_bin_100"] == 0]
    bin1_large_buy = bin1.groupby(["StockId", "Date_int"])["_large_buy"].sum().reset_index()
    bin1_large_buy.columns = ["StockId", "Date_int", "raw_bin1_large_buy"]
    
    # 計算 loss_domain × risk_seeking (per bin)
    # 需要每個 bin 的小單價格標準差和均值
    # 使用 Series 的 loc 索引避免長度不匹配的問題
    small_ticks_mask = tick_data["_small_sell"] > 0
    small_ticks = tick_data[small_ticks_mask].copy()
    if not small_ticks.empty:
        small_stats = small_ticks.groupby(["StockId", "Date_int", "_bin_100"]).agg(
            sp_std=("DealPrice", "std"),
            sp_mean=("DealPrice", "mean"),
        ).reset_index()
        bin_agg = bin_agg.merge(small_stats, on=["StockId", "Date_int", "_bin_100"], how="left")
        bin_agg["sp_std"] = bin_agg["sp_std"].fillna(0)
        bin_agg["sp_mean"] = bin_agg["sp_mean"].fillna(1)
        bin_agg["risk_seeking"] = bin_agg["sp_std"] / (bin_agg["sp_mean"] + 1e-8)
    else:
        bin_agg["risk_seeking"] = 0
    
    # 合併開盤價來計算 loss_domain
    open_prices = df[["StockId", "Date", "開盤價"]].copy()
    open_prices["Date_int"] = open_prices["Date"].astype(int)
    open_prices = open_prices.rename(columns={"Date_int": "Date_int2", "開盤價": "open_p"})
    open_prices = open_prices.drop(columns=["Date"])
    open_prices = open_prices.rename(columns={"Date_int2": "Date_int"})
    
    # 合併開盤價到 bin_agg
    bin_agg = bin_agg.merge(open_prices, on=["StockId", "Date_int"], how="left")
    bin_agg["open_p"] = bin_agg["open_p"].fillna(bin_agg["bin_vwap"])
    bin_agg["loss_domain"] = (bin_agg["bin_vwap"] < bin_agg["open_p"] * 0.98).astype(int)
    bin_agg["raw_per_bin"] = bin_agg["loss_domain"] * bin_agg["risk_seeking"]
    
    # 取每日最大值
    loss_domain_max = bin_agg.groupby(["StockId", "Date_int"])["raw_per_bin"].max().reset_index()
    loss_domain_max.columns = ["StockId", "Date_int", "raw_loss_domain_max"]
    
    # 計算前一日收盤價
    df_sorted = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
    df_sorted["_prev_close"] = df_sorted.groupby("StockId")["收盤價"].shift(1)
    
    # neg_mood: open < prev_close * 0.97
    df_sorted["raw_neg_mood"] = (
        (df_sorted["開盤價"] < df_sorted["_prev_close"] * 0.97)
    ).fillna(0).astype(int)
    
    # gap_down: open < prev_close * 0.98
    df_sorted["raw_gap_down"] = (
        (df_sorted["開盤價"] < df_sorted["_prev_close"] * 0.98)
    ).fillna(0).astype(int)
    
    # high_prior: close_now/close_20 - 1 > 0.15
    df_sorted["_close_20"] = df_sorted.groupby("StockId")["收盤價"].shift(20)
    df_sorted["raw_high_prior"] = (
        (df_sorted["收盤價"] / df_sorted["_close_20"] - 1 > 0.15)
    ).fillna(0).astype(int)
    
    # 提取需要合併的欄位
    gap_cols = df_sorted[["StockId", "Date", "raw_neg_mood", "raw_gap_down", "raw_high_prior"]].copy()
    
    # 合併所有 intermediates 到 df
    mood_sell = mood_sell.rename(columns={"Date_int": "Date"})
    instant_exit = instant_exit.rename(columns={"Date_int": "Date"})
    bin1_large_buy = bin1_large_buy.rename(columns={"Date_int": "Date"})
    loss_domain_max = loss_domain_max.rename(columns={"Date_int": "Date"})
    
    df = df.merge(gap_cols, on=["StockId", "Date"], how="left")
    df = df.merge(mood_sell[["StockId", "Date", "raw_mood_sell"]], on=["StockId", "Date"], how="left")
    df = df.merge(instant_exit[["StockId", "Date", "raw_instant_exit"]], on=["StockId", "Date"], how="left")
    df = df.merge(bin1_large_buy[["StockId", "Date", "raw_bin1_large_buy"]], on=["StockId", "Date"], how="left")
    df = df.merge(loss_domain_max[["StockId", "Date", "raw_loss_domain_max"]], on=["StockId", "Date"], how="left")
    
    df["raw_neg_mood"] = df["raw_neg_mood"].fillna(0)
    df["raw_gap_down"] = df["raw_gap_down"].fillna(0)
    df["raw_high_prior"] = df["raw_high_prior"].fillna(0)
    df["raw_mood_sell"] = df["raw_mood_sell"].fillna(0)
    df["raw_instant_exit"] = df["raw_instant_exit"].fillna(0)
    df["raw_bin1_large_buy"] = df["raw_bin1_large_buy"].fillna(0)
    df["raw_loss_domain_max"] = df["raw_loss_domain_max"].fillna(0)
    
    return df


def _compute_small_buy_ticks(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    small_buy_mask = (tick_data["DealCount"] == 1) & (tick_data["PrFlag"] == 1)
    small_buy_daily = tick_data[small_buy_mask].groupby(["StockId", "Date_int"]).size().reset_index()
    small_buy_daily.columns = ["StockId", "Date_int", "raw_small_buy_ticks"]
    small_buy_daily = small_buy_daily.rename(columns={"Date_int": "Date"})
    
    df = df.merge(small_buy_daily[["StockId", "Date", "raw_small_buy_ticks"]], on=["StockId", "Date"], how="left")
    df["raw_small_buy_ticks"] = df["raw_small_buy_ticks"].fillna(0)
    
    return df


def _compute_first500_small_buy(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    if "Small_Thresh" not in tick_data.columns and "Small_Thresh" in df.columns:
        thresh_df = df[["StockId", "Date", "Small_Thresh"]].copy()
        thresh_df["Date_int"] = thresh_df["Date"].astype(int)
        tick_data = tick_data.merge(thresh_df[["StockId", "Date_int", "Small_Thresh"]], on=["StockId", "Date_int"], how="left")
        tick_data["Small_Thresh"] = tick_data["Small_Thresh"].fillna(1)
    
    first500 = tick_data.groupby(["StockId", "Date_int"]).head(500)
    small_thresh_val = tick_data["Small_Thresh"].values if "Small_Thresh" in tick_data.columns else 1
    first500["_small_active_buy"] = (
        (first500["PrFlag"] == 1) & 
        (first500["DealCount"].values <= small_thresh_val[:len(first500)])
    ).astype(int) * first500["DealCount"]
    
    result = first500.groupby(["StockId", "Date_int"])["_small_active_buy"].sum().reset_index()
    result.columns = ["StockId", "Date_int", "raw_first500_small_buy"]
    result = result.rename(columns={"Date_int": "Date"})
    
    df = df.merge(result[["StockId", "Date", "raw_first500_small_buy"]], on=["StockId", "Date"], how="left")
    df["raw_first500_small_buy"] = df["raw_first500_small_buy"].fillna(0)
    
    return df


def _compute_last500_small_sell(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    if "Small_Thresh" not in tick_data.columns and "Small_Thresh" in df.columns:
        thresh_df = df[["StockId", "Date", "Small_Thresh"]].copy()
        thresh_df["Date_int"] = thresh_df["Date"].astype(int)
        tick_data = tick_data.merge(thresh_df[["StockId", "Date_int", "Small_Thresh"]], on=["StockId", "Date_int"], how="left")
        tick_data["Small_Thresh"] = tick_data["Small_Thresh"].fillna(1)
    
    tick_data["_rank_rev"] = tick_data.groupby(["StockId", "Date_int"]).cumcount(ascending=False)
    last500 = tick_data[tick_data["_rank_rev"] < 500]
    small_thresh_val = tick_data["Small_Thresh"].values if "Small_Thresh" in tick_data.columns else 1
    
    last500["_small_active_sell"] = (
        (last500["PrFlag"] == 0) & 
        (last500["DealCount"].values <= small_thresh_val[:len(last500)])
    ).astype(int) * last500["DealCount"]
    
    result = last500.groupby(["StockId", "Date_int"])["_small_active_sell"].sum().reset_index()
    result.columns = ["StockId", "Date_int", "raw_last500_small_sell"]
    result = result.rename(columns={"Date_int": "Date"})
    
    df = df.merge(result[["StockId", "Date", "raw_last500_small_sell"]], on=["StockId", "Date"], how="left")
    df["raw_last500_small_sell"] = df["raw_last500_small_sell"].fillna(0)
    
    return df


def _compute_bin_aggs_200(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    tick_data["_pv"] = tick_data["DealPrice"] * tick_data["DealCount"]
    tick_data["_is_buy"] = (tick_data["PrFlag"] == 1).astype(int)
    tick_data["_is_sell"] = (tick_data["PrFlag"] == 0).astype(int)
    
    tick_data["_bin_200"] = _make_bins_vectorized(tick_data, 200)
    valid = tick_data[tick_data["_bin_200"] >= 0].copy()
    
    if valid.empty:
        return df
    
    bin_agg = valid.groupby(["StockId", "Date_int", "_bin_200"]).agg(
        bin_vwap_num=("_pv", "sum"),
        bin_vwap_den=("DealCount", "sum"),
        buy_ticks=("_is_buy", "sum"),
        sell_ticks=("_is_sell", "sum"),
        total_vol=("DealCount", "sum"),
    ).reset_index()
    
    bin_agg["bin_vwap"] = bin_agg["bin_vwap_num"] / (bin_agg["bin_vwap_den"] + 1e-8)
    bin_agg = bin_agg.sort_values(["StockId", "Date_int", "_bin_200"]).reset_index(drop=True)
    bin_agg["prev_vwap"] = bin_agg.groupby(["StockId", "Date_int"])["bin_vwap"].shift(1)
    bin_agg["prev_vwap"] = bin_agg["prev_vwap"].fillna(bin_agg["bin_vwap"])
    
    bin_agg["garbling"] = 1.0 / (abs(bin_agg["buy_ticks"] - bin_agg["sell_ticks"]) + 1e-5)
    bin_agg["dumping"] = (bin_agg["bin_vwap"] < bin_agg["prev_vwap"]).astype(int) * bin_agg["total_vol"]
    bin_agg["raw_per_bin"] = bin_agg["garbling"] * bin_agg["dumping"]
    
    daily_raw = bin_agg.groupby(["StockId", "Date_int"])["raw_per_bin"].sum().reset_index()
    daily_raw.columns = ["StockId", "Date_int", "raw_bin200_garbling_dumping"]
    daily_raw = daily_raw.rename(columns={"Date_int": "Date"})
    
    df = df.merge(daily_raw[["StockId", "Date", "raw_bin200_garbling_dumping"]], on=["StockId", "Date"], how="left")
    df["raw_bin200_garbling_dumping"] = df["raw_bin200_garbling_dumping"].fillna(0)
    
    return df


def _compute_bin_aggs_50_extended(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    if "Small_Thresh" not in tick_data.columns and "Small_Thresh" in df.columns:
        thresh_df = df[["StockId", "Date", "Small_Thresh", "Large_Thresh"]].copy()
        thresh_df["Date_int"] = thresh_df["Date"].astype(int)
        tick_data = tick_data.merge(thresh_df[["StockId", "Date_int", "Small_Thresh", "Large_Thresh"]], on=["StockId", "Date_int"], how="left")
        tick_data["Small_Thresh"] = tick_data["Small_Thresh"].fillna(1)
        tick_data["Large_Thresh"] = tick_data["Large_Thresh"].fillna(1)
    
    tick_data["_pv"] = tick_data["DealPrice"] * tick_data["DealCount"]
    tick_data["_is_buy"] = (tick_data["PrFlag"] == 1).astype(int)
    tick_data["_is_sell"] = (tick_data["PrFlag"] == 0).astype(int)
    
    large_thresh_val = tick_data["Large_Thresh"].values if "Large_Thresh" in tick_data.columns else 1
    tick_data["_large_buy"] = (
        (tick_data["_is_buy"] == 1) & 
        (tick_data["DealCount"].values >= large_thresh_val)
    ).astype(int) * tick_data["DealCount"]
    tick_data["_sell_vol"] = tick_data["DealCount"] * tick_data["_is_sell"]
    
    tick_data["_bin_50"] = _make_bins_vectorized(tick_data, 50)
    valid = tick_data[tick_data["_bin_50"] >= 0].copy()
    
    if valid.empty:
        return df
    
    bin_agg = valid.groupby(["StockId", "Date_int", "_bin_50"]).agg(
        bin_high=("DealPrice", "max"),
        bin_low=("DealPrice", "min"),
        bin_open=("DealPrice", "first"),
        bin_vwap_num=("_pv", "sum"),
        bin_vwap_den=("DealCount", "sum"),
        large_buy=("_large_buy", "sum"),
        sell_vol=("_sell_vol", "sum"),
    ).reset_index()
    
    open_prices = df[["StockId", "Date", "開盤價"]].copy()
    open_prices["Date_int"] = open_prices["Date"].astype(int)
    open_prices = open_prices.rename(columns={"Date_int": "Date_int2", "開盤價": "open_p"})
    open_prices = open_prices.drop(columns=["Date"]).rename(columns={"Date_int2": "Date_int"})
    bin_agg = bin_agg.merge(open_prices, on=["StockId", "Date_int"], how="left")
    bin_agg["open_p"] = bin_agg["open_p"].fillna(bin_agg["bin_vwap_num"] / (bin_agg["bin_vwap_den"] + 1e-8))
    
    bin_agg["bin_vwap"] = bin_agg["bin_vwap_num"] / (bin_agg["bin_vwap_den"] + 1e-8)
    bin_agg["salience"] = (bin_agg["bin_high"] - bin_agg["bin_low"]) / (bin_agg["open_p"] + 1e-5)
    
    day_avg_vol = bin_agg.groupby(["StockId", "Date_int"])["bin_vwap_den"].transform("mean")
    bin_agg["vol_quality"] = bin_agg["bin_vwap_den"] / (day_avg_vol + 1e-5)
    bin_agg["trap"] = bin_agg["salience"] / (bin_agg["vol_quality"] + 1e-5)
    
    salience_max = bin_agg.groupby(["StockId", "Date_int"])["trap"].max().reset_index()
    salience_max.columns = ["StockId", "Date_int", "raw_salience_trap_max"]
    
    bin_agg["loss_domain"] = (bin_agg["bin_vwap"] < bin_agg["open_p"] * 0.98).astype(int)
    bin_agg["volatility"] = (bin_agg["bin_high"] - bin_agg["bin_low"]) / (bin_agg["open_p"] + 1e-8)
    bin_agg["raw_per_bin_45"] = bin_agg["loss_domain"] * bin_agg["volatility"] * bin_agg["sell_vol"]
    
    loss_max = bin_agg.groupby(["StockId", "Date_int"])["raw_per_bin_45"].max().reset_index()
    loss_max.columns = ["StockId", "Date_int", "raw_loss_domain_vol_max"]
    
    bin1 = valid[valid["_bin_50"] == 0]
    bin1_large_buy = bin1.groupby(["StockId", "Date_int"])["_large_buy"].sum().reset_index()
    bin1_large_buy.columns = ["StockId", "Date_int", "raw_bin50_bin1_large_buy"]
    
    total_large_buy = valid.groupby(["StockId", "Date_int"])["_large_buy"].sum().reset_index()
    total_large_buy.columns = ["StockId", "Date_int", "raw_bin50_total_large_buy"]
    
    salience_max = salience_max.rename(columns={"Date_int": "Date"})
    loss_max = loss_max.rename(columns={"Date_int": "Date"})
    bin1_large_buy = bin1_large_buy.rename(columns={"Date_int": "Date"})
    total_large_buy = total_large_buy.rename(columns={"Date_int": "Date"})
    
    df = df.merge(salience_max[["StockId", "Date", "raw_salience_trap_max"]], on=["StockId", "Date"], how="left")
    df = df.merge(loss_max[["StockId", "Date", "raw_loss_domain_vol_max"]], on=["StockId", "Date"], how="left")
    df = df.merge(bin1_large_buy[["StockId", "Date", "raw_bin50_bin1_large_buy"]], on=["StockId", "Date"], how="left")
    df = df.merge(total_large_buy[["StockId", "Date", "raw_bin50_total_large_buy"]], on=["StockId", "Date"], how="left")
    
    df["raw_salience_trap_max"] = df["raw_salience_trap_max"].fillna(0)
    df["raw_loss_domain_vol_max"] = df["raw_loss_domain_vol_max"].fillna(0)
    df["raw_bin50_bin1_large_buy"] = df["raw_bin50_bin1_large_buy"].fillna(0)
    df["raw_bin50_total_large_buy"] = df["raw_bin50_total_large_buy"].fillna(0)
    
    return df


def _compute_vol_surface(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
    
    close = df["收盤價"].astype(float)
    ret1 = close.pct_change(1)
    df["raw_ret1"] = ret1
    
    df["raw_vol20"] = ret1.rolling(20, min_periods=10).std()
    df["raw_vol20_q20"] = df["raw_vol20"].rolling(20, min_periods=10).quantile(0.2)
    
    df["raw_high_shift1"] = df.groupby("StockId")["最高價"].shift(1)
    df["raw_high_shift1"] = df["raw_high_shift1"].fillna(0)
    
    df["raw_ret1_shift1"] = df.groupby("StockId")["raw_ret1"].shift(1)
    df["raw_ret1_shift1"] = df["raw_ret1_shift1"].fillna(0)
    
    return df


def _compute_ret_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
    
    close = df["收盤價"].astype(float)
    open_p = df["開盤價"].astype(float)
    prev_close = df.groupby("StockId")["收盤價"].shift(1)
    
    df["raw_gap_down"] = ((open_p < prev_close * 0.98)).fillna(0).astype(int)
    
    ret5 = close.pct_change(5)
    df["raw_ret5"] = ret5.fillna(0)
    
    return df


def _compute_screening_premium(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
    
    open_p = df["開盤價"].astype(float)
    prev_high = df.groupby("StockId")["最高價"].shift(1).astype(float)
    
    df["raw_screening_premium"] = np.where(
        prev_high > 0,
        (open_p - prev_high) / (prev_high + 1e-8),
        0.0
    )
    
    return df


def _compute_afd_intermediates(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    tick_data["_pv"] = tick_data["DealPrice"] * tick_data["DealCount"]
    tick_data["_is_buy"] = (tick_data["PrFlag"] == 1).astype(int)
    
    daily_agg = tick_data.groupby(["StockId", "Date_int"]).agg(
        _total_vol=("DealCount", "sum"),
        _total_value=("_pv", "sum"),
    ).reset_index()
    
    daily_agg["vwap"] = daily_agg["_total_value"] / (daily_agg["_total_vol"] + 1e-10)
    
    def _calc_tick_slope(grp):
        prices = grp["DealPrice"].values
        if len(prices) > 2:
            x = np.arange(len(prices), dtype=np.float64)
            return np.polyfit(x, prices, 1)[0]
        return 0.0
    
    tick_slope_agg = tick_data.groupby(["StockId", "Date_int"]).apply(_calc_tick_slope).reset_index()
    tick_slope_agg.columns = ["StockId", "Date_int", "tick_slope"]
    
    tick_data = tick_data.merge(daily_agg[["StockId", "Date_int", "vwap"]], on=["StockId", "Date_int"], how="left")
    tick_data = tick_data.merge(tick_slope_agg[["StockId", "Date_int", "tick_slope"]], on=["StockId", "Date_int"], how="left")
    
    tick_data["_cold_zone"] = (tick_data["DealPrice"] < tick_data["vwap"]).astype(int)
    tick_data["_cold_zone_buy"] = tick_data["_cold_zone"] * tick_data["_is_buy"] * tick_data["DealCount"]
    
    cold_buy_agg = tick_data.groupby(["StockId", "Date_int"])["_cold_zone_buy"].sum().reset_index()
    cold_buy_agg.columns = ["StockId", "Date_int", "raw_cold_zone_buy"]
    
    def _calc_cold_slope(grp):
        cold_prices = grp[grp["_cold_zone"] == 1]["DealPrice"].values
        if len(cold_prices) > 5:
            x = np.arange(len(cold_prices), dtype=np.float64)
            return np.polyfit(x, cold_prices, 1)[0]
        return 0.0
    
    cold_slope_agg = tick_data.groupby(["StockId", "Date_int"]).apply(_calc_cold_slope).reset_index()
    cold_slope_agg.columns = ["StockId", "Date_int", "raw_cold_slope"]
    
    tick_mean_price_agg = tick_data.groupby(["StockId", "Date_int"])["DealPrice"].mean().reset_index()
    tick_mean_price_agg.columns = ["StockId", "Date_int", "raw_tick_mean_price"]
    
    tick_data["_above_vwap_buy"] = np.where(
        (tick_data["DealPrice"] > tick_data["vwap"] * 0.98) & (tick_data["PrFlag"] == 1),
        tick_data["DealCount"], 0
    )
    tick_data["_above_vwap_high_buy"] = np.where(
        (tick_data["DealPrice"] > tick_data["vwap"] * 1.02) & (tick_data["PrFlag"] == 1),
        tick_data["DealCount"], 0
    )
    
    above_vwap_buy_agg = tick_data.groupby(["StockId", "Date_int"])["_above_vwap_buy"].sum().reset_index()
    above_vwap_buy_agg.columns = ["StockId", "Date_int", "above_vwap_buy"]
    above_vwap_high_buy_agg = tick_data.groupby(["StockId", "Date_int"])["_above_vwap_high_buy"].sum().reset_index()
    above_vwap_high_buy_agg.columns = ["StockId", "Date_int", "above_vwap_high_buy"]
    
    def _calc_q85_buy(grp):
        dc = grp["DealCount"].values
        pf = grp["PrFlag"].values
        n = len(dc)
        q85 = np.percentile(dc, 85) if n > 10 else dc.max()
        large_buy = np.sum(dc[(dc >= q85) & (pf == 1)])
        small_buy = np.sum(dc[(dc < q85) & (pf == 1)])
        return pd.Series({"q85_large_buy": large_buy, "q85_small_buy": small_buy})
    
    q85_buy_agg = tick_data.groupby(["StockId", "Date_int"]).apply(_calc_q85_buy).reset_index()
    q85_buy_agg.columns = ["StockId", "Date_int", "q85_large_buy", "q85_small_buy"]
    q85_buy_agg = q85_buy_agg.rename(columns={"Date_int": "Date"})
    
    df = df.merge(q85_buy_agg[["StockId", "Date", "q85_large_buy", "q85_small_buy"]], on=["StockId", "Date"], how="left")
    df["q85_large_buy"] = df["q85_large_buy"].fillna(0)
    df["q85_small_buy"] = df["q85_small_buy"].fillna(0)
    
    cold_buy_agg = cold_buy_agg.rename(columns={"Date_int": "Date"})
    cold_slope_agg = cold_slope_agg.rename(columns={"Date_int": "Date"})
    tick_mean_price_agg = tick_mean_price_agg.rename(columns={"Date_int": "Date"})
    above_vwap_buy_agg = above_vwap_buy_agg.rename(columns={"Date_int": "Date"})
    above_vwap_high_buy_agg = above_vwap_high_buy_agg.rename(columns={"Date_int": "Date"})
    
    df = df.merge(cold_buy_agg[["StockId", "Date", "raw_cold_zone_buy"]], on=["StockId", "Date"], how="left")
    df = df.merge(cold_slope_agg[["StockId", "Date", "raw_cold_slope"]], on=["StockId", "Date"], how="left")
    df = df.merge(tick_mean_price_agg[["StockId", "Date", "raw_tick_mean_price"]], on=["StockId", "Date"], how="left")
    df = df.merge(above_vwap_buy_agg[["StockId", "Date", "above_vwap_buy"]], on=["StockId", "Date"], how="left")
    df = df.merge(above_vwap_high_buy_agg[["StockId", "Date", "above_vwap_high_buy"]], on=["StockId", "Date"], how="left")
    
    tick_slope_for_merge = tick_slope_agg.rename(columns={"Date_int": "Date"})
    df = df.merge(tick_slope_for_merge[["StockId", "Date", "tick_slope"]], on=["StockId", "Date"], how="left")
    
    df["raw_cold_zone_buy"] = df["raw_cold_zone_buy"].fillna(0)
    df["raw_cold_slope"] = df["raw_cold_slope"].fillna(0)
    df["raw_tick_mean_price"] = df["raw_tick_mean_price"].fillna(df["vwap"])
    df["above_vwap_buy"] = df["above_vwap_buy"].fillna(0)
    df["above_vwap_high_buy"] = df["above_vwap_high_buy"].fillna(0)
    df["tick_slope"] = df["tick_slope"].fillna(0)
    
    if "_small_buy" not in df.columns:
        df["_small_buy"] = 0
    if "_large_buy" not in df.columns:
        df["_large_buy"] = 0
    
    return df


def _merge_margin_balance(df_daily: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    margin_col = "融券餘額(張)"
    if margin_col in df.columns:
        df["raw_margin_balance"] = df[margin_col].fillna(0)
    else:
        df["raw_margin_balance"] = 0
    return df


def _compute_afd_tick_level_intermediates(tick_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    tick_data = tick_data.copy()
    tick_data["Date_int"] = tick_data["Date"].astype(int)
    tick_data = tick_data.sort_values(["StockId", "Date_int", "TotalQty"]).reset_index(drop=True)
    
    has_spread = "BuyPr" in tick_data.columns and "SellPr" in tick_data.columns
    if not has_spread:
        df["raw_optimal_drag_raw"] = 0
        df["raw_friction_riverbend_raw"] = 0
        return df
    
    f49_records = []
    f50_records = []
    
    for (stock_id, date_int), grp in tick_data.groupby(["StockId", "Date_int"]):
        if len(grp) < 10:
            continue
        
        dp = grp["DealPrice"].values.astype(np.float64)
        dc = grp["DealCount"].values.astype(np.float64)
        sp = grp["SellPr"].values
        bp = grp["BuyPr"].values
        pf = grp["PrFlag"].values
        n = len(dp)
        
        spread_mean = np.mean(sp - bp)
        vol = np.sum(dc)
        drag_ratio = spread_mean / (vol + 1e-10)
        
        x = np.arange(len(dp), dtype=np.float64)
        slope = np.polyfit(x, dp, 1)[0] if len(dp) > 2 else 0.0
        
        f49_records.append({
            "StockId": stock_id,
            "Date": date_int,
            "raw_drag_ratio": drag_ratio,
            "raw_tick_slope": slope,
        })
        
        BIN = 50
        n_bins = n // BIN
        total_val = 0.0
        for i in range(n_bins):
            s, e = i * BIN, (i + 1) * BIN
            spread = sp[s:e] - bp[s:e]
            v_cross = np.mean(np.diff(spread[:min(5, len(spread))])) if len(spread) > 1 else 0.0
            direction = np.where(pf[s:e] == 1, 1.0, np.where(pf[s:e] == 0, -1.0, 0.0))
            v_stream = np.sum(dc[s:e] * direction)
            delta_p = max(0, dp[e - 1] - dp[s])
            total_val += v_cross * v_stream * delta_p
        
        f50_records.append({
            "StockId": stock_id,
            "Date": date_int,
            "raw_riverbend_val": total_val,
        })
    
    if f49_records:
        f49_df = pd.DataFrame(f49_records)
        df = df.merge(f49_df[["StockId", "Date", "raw_drag_ratio", "raw_tick_slope"]], 
                      on=["StockId", "Date"], how="left")
    else:
        df["raw_drag_ratio"] = 0
        df["raw_tick_slope"] = 0
    
    if f50_records:
        f50_df = pd.DataFrame(f50_records)
        df = df.merge(f50_df[["StockId", "Date", "raw_riverbend_val"]], 
                      on=["StockId", "Date"], how="left")
    else:
        df["raw_riverbend_val"] = 0
    
    df["raw_drag_ratio"] = df["raw_drag_ratio"].fillna(0)
    df["raw_tick_slope"] = df["raw_tick_slope"].fillna(0)
    df["raw_riverbend_val"] = df["raw_riverbend_val"].fillna(0)
    
    return df
