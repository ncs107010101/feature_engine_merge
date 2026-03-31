import pandas as pd
import numpy as np
from .single_stock_broker import preprocess_single_stock_broker


def preprocess_cross_broker_daily(data: pd.DataFrame, return_broker_day: bool = False, **kwargs) -> pd.DataFrame:
    if "daily_data" not in kwargs:
        raise ValueError(
            "cross_broker_daily requires daily OHLC data passed as "
            "`daily_data` in kwargs."
        )

    df_daily = kwargs["daily_data"].copy()
    tick_intermediates = kwargs.get("tick_intermediates", None)
    
    req_daily = ["StockId", "Date", "收盤價", "最高價", "最低價", "成交量(千股)", "報酬率", "開盤價"]
    missing = [c for c in req_daily if c not in df_daily.columns]
    if missing:
        raise ValueError(
            f"daily_data is missing required columns: {missing}. "
            "Note: PHY features require '收盤價', '報酬率', and '成交量(千股)'."
        )

    broker_daily, broker_day_original = preprocess_single_stock_broker(data, return_broker_day=True)

    # --- Step 2: Start with daily data (all dates), left-join broker aggregations ---
    df_daily["Date"] = df_daily["Date"].astype(int)
    broker_daily["Date"] = broker_daily["Date"].astype(int)

    merge_cols = ["StockId", "Date"]
    daily_cols_needed = ["收盤價", "最高價", "最低價", "成交量(千股)", "報酬率", "開盤價"]
    if "成交金額(元)" in df_daily.columns:
        daily_cols_needed.append("成交金額(元)")

    # Start with daily data as base (all trading dates), left-join broker data
    _df = df_daily[merge_cols + daily_cols_needed].copy()
    _df = _df.sort_values(["StockId", "Date"])

    # Left-join ALL broker daily columns into daily (drop duplicate cols from broker_daily)
    broker_daily_for_merge = broker_daily.drop(columns=[c for c in broker_daily.columns if c in daily_cols_needed and c not in ["DailyVWAP"]], errors="ignore")
    _df = _df.merge(broker_daily_for_merge, on=merge_cols, how="left")
    
    if tick_intermediates is not None:
        tick_int = tick_intermediates[["StockId", "Date", "raw_small_buy_ticks"]].copy()
        tick_int["Date"] = tick_int["Date"].astype(int)
        _df = _df.merge(tick_int, on=merge_cols, how="left")
        _df["raw_small_buy_ticks"] = _df["raw_small_buy_ticks"].fillna(0)

    # --- Step 3: Compute shared intermediates ---
    _df = _df.sort_values(["StockId", "Date"])

    # ATR-20 (shared by f_retail_trap and f_cost_stack)
    prev_close = _df.groupby("StockId")["收盤價"].shift(1)
    tr1 = _df["最高價"] - _df["最低價"]
    tr2 = (_df["最高價"] - prev_close).abs()
    tr3 = (_df["最低價"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    _df["_atr_20"] = (
        tr.groupby(_df["StockId"])
        .rolling(window=20, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Daily return (shared by f_flow_deviation)
    _df["_daily_return"] = (
        _df.groupby("StockId")["收盤價"]
        .pct_change()
        .fillna(0.0)
    )

    # === PHY Alpha v1: 8 broker + daily features ===
    # All these features require broker data + daily OHLC.
    
    # ---- Step A: Broker-side aggregations shared across features ----
    _brk_raw = data.copy()
    _brk_raw["Date"] = _brk_raw["Date"].astype(int)

    _brk_day = _brk_raw.groupby(["StockId", "Date", "BrokerId"]).agg(
        BuyQtm=("BuyQtm", "sum"),
        SellQtm=("SellQtm", "sum"),
    ).reset_index()
    _brk_day["TotalQtm"] = _brk_day["BuyQtm"] + _brk_day["SellQtm"]

    _brk_day["RankPct"] = (
        _brk_day.groupby(["StockId", "Date"])["TotalQtm"]
        .rank(pct=True, method="first", ascending=False)
    )
    _brk_day["RankAbs"] = (
        _brk_day.groupby(["StockId", "Date"])["TotalQtm"]
        .rank(method="first", ascending=False)
    )

    _is_top10 = _brk_day["RankPct"] <= 0.1
    _is_bot80 = _brk_day["RankPct"] >= 0.2
    _is_top5  = _brk_day["RankAbs"]  <= 5

    def _net_buy_agg(mask, col="NetBuy"):
        sub = _brk_day[mask].copy()
        sub["NetBuy"] = sub["BuyQtm"] - sub["SellQtm"]
        return sub.groupby(["StockId", "Date"])["NetBuy"].sum().rename(col)

    _top10_net = _net_buy_agg(_is_top10, "Top10_NetBuy")
    _bot80_net = _net_buy_agg(_is_bot80, "Bot80_NetBuy")
    _top5_net  = _net_buy_agg(_is_top5,  "Top5_NetBuy")

    _brk_day["BuyRank"] = _brk_day.groupby(["StockId", "Date"])["BuyQtm"].rank(method="first", ascending=False)
    _brk_day["SellRank"] = _brk_day.groupby(["StockId", "Date"])["SellQtm"].rank(method="first", ascending=False)
    
    _top5_buy = (
        _brk_day[_brk_day["BuyRank"] <= 5]
        .groupby(["StockId", "Date"])["BuyQtm"].sum().rename("Top5_Buy")
    )
    _top5_sell = (
        _brk_day[_brk_day["SellRank"] <= 5]
        .groupby(["StockId", "Date"])["SellQtm"].sum().rename("Top5_Sell")
    )

    # ---- Step B: Set index for alignment ----
    _df = _df.set_index(["StockId", "Date"])
    _daily_vol_shares = _df["成交量(千股)"] * 1000
    _daily_vol_safe   = np.where(_daily_vol_shares == 0, 1.0, _daily_vol_shares)

    _top10_nbr = (_top10_net * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe
    _bot80_nbr = (_bot80_net * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe
    _top5_nbr  = (_top5_net  * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe
    _top5_brat = (_top5_buy  * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe
    _top5_srat = (_top5_sell * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe

    _close = _df["收盤價"]
    _ret   = _df["報酬率"]
    _h_d   = _df["最高價"]
    _l_d   = _df["最低價"]

    # ---- Step C: Rolling 20-day correlations ----
    def _rolling_corr20(s1: pd.Series, s2: pd.Series) -> pd.Series:
        out = pd.Series(index=s1.index, dtype=float)
        for sid in s1.index.get_level_values("StockId").unique():
            g1 = s1.xs(sid, level="StockId")
            g2 = s2.xs(sid, level="StockId")
            # Handle cases with few data points
            if len(g1) >= 5:
                corr = g1.rolling(20, min_periods=5).corr(g2).fillna(0.0)
                out.loc[(sid,)] = corr.values
            else:
                out.loc[(sid,)] = 0.0
        return out

    _corr_bright   = _rolling_corr20(_top10_nbr, _bot80_nbr)
    _corr_entangle = _rolling_corr20(_top5_nbr,  _ret)

    # ---- Step D: Score for multiplex_coherence ----
    def _multi_score(close_s: pd.Series) -> pd.Series:
        r1  = close_s.pct_change(1).fillna(0.0)
        r5  = close_s.pct_change(5).fillna(0.0)
        r10 = close_s.pct_change(10).fillna(0.0)
        r20 = close_s.pct_change(20).fillna(0.0)
        return (
            np.sign(r1) * np.abs(r1)
            + np.sign(r5)  * np.abs(r5)  / 5.0
            + np.sign(r10) * np.abs(r10) / 10.0
            + np.sign(r20) * np.abs(r20) / 20.0
        )

    _score = pd.Series(index=_df.index, dtype=float)
    for _sid in _df.index.get_level_values("StockId").unique():
        _c = _close.xs(_sid, level="StockId")
        _score.loc[(_sid,)] = _multi_score(_c).values

    # ---- Step E: Geometric torque ----
    _h20 = (
        _h_d.groupby(level="StockId")
        .transform(lambda x: x.rolling(20, min_periods=1).max())
    )
    _l20 = (
        _l_d.groupby(level="StockId")
        .transform(lambda x: x.rolling(20, min_periods=1).min())
    )
    _dist_h = np.abs(_close - _h20) / (_h20 + 1e-6)
    _dist_l = np.abs(_close - _l20) / (_l20 + 1e-6)
    _G_torque = 1.0 / ((_dist_h + _dist_l) + 0.01)

    # ---- Step F: Interstitial stress position ----
    _pos = ((_close - _l_d) / (_h_d - _l_d + 1e-6)).clip(0, 1)

    # ---- Step G: Supernova VWAP deviation ----
    if "成交金額(元)" in _df.columns:
        _daily_vwap = _df["成交金額(元)"] / (_daily_vol_shares + 1e-6)
        _daily_vwap = np.where(_daily_vwap == 0, _close.values, _daily_vwap)
    else:
        _daily_vwap = _close.values
    _price_dev = np.abs(_close.values / (_daily_vwap + 1e-6) - 1.0)

    # ---- Step H: Geodesic VWAP cost advantage ----
    _has_amounts = (
        "BuyAmount" in _brk_raw.columns and "SellAmount" in _brk_raw.columns
    )
    if _has_amounts:
        _brk_day2 = _brk_raw.groupby(["StockId", "Date", "BrokerId"]).agg(
            BuyQtm=("BuyQtm", "sum"),
            SellQtm=("SellQtm", "sum"),
            BuyAmt=("BuyAmount", "sum"),
            SellAmt=("SellAmount", "sum"),
        ).reset_index()
        _brk_day2["TotalQtm"] = _brk_day2["BuyQtm"] + _brk_day2["SellQtm"]
        _brk_day2["RankAbs"] = (
            _brk_day2.groupby(["StockId", "Date"])["TotalQtm"]
            .rank(method="first", ascending=False)
        )
        _t5 = _brk_day2[_brk_day2["RankAbs"] <= 5]
        _rt = _brk_day2[_brk_day2["RankAbs"] > 5]
        def _vwap_g(sub, qty_col, amt_col):
            num = sub.groupby(["StockId", "Date"])[amt_col].sum()
            den = sub.groupby(["StockId", "Date"])[qty_col].sum() * 1000
            return (num / (den + 1e-6)).reindex(_df.index).fillna(0.0)

        _t5_bvwap  = _vwap_g(_t5, "BuyQtm",  "BuyAmt")
        _t5_svwap  = _vwap_g(_t5, "SellQtm", "SellAmt")
        _rt_bvwap  = _vwap_g(_rt, "BuyQtm",  "BuyAmt")
        _rt_svwap  = _vwap_g(_rt, "SellQtm", "SellAmt")
        _top5_nb   = (_top5_net * 1000).reindex(_df.index).fillna(0.0)
        _cost_adv  = np.where(
            _top5_nb > 0, _rt_bvwap - _t5_bvwap,
            np.where(_top5_nb < 0, _t5_svwap - _rt_svwap, 0.0),
        )
        _cost_adv_ratio = pd.Series(_cost_adv, index=_df.index) / (_close + 1e-6)
        _geodesic = pd.Series(
            (np.sign(_top5_nbr) * np.abs(_cost_adv_ratio * _top5_nbr)),
            index=_df.index
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        _geodesic = pd.Series((_top5_nbr * _ret), index=_df.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ---- Step I: Write columns ----
    _df["raw_phy_bright_dark_mode_covariance"] = (-1.0 * _corr_bright * _top10_nbr).fillna(0.0)
    _df["raw_phy_dark_em_repulsive_force"] = (-1.0 * (_bot80_nbr - _top10_nbr) * _ret.abs()).fillna(0.0)
    _df["raw_phy_entanglement_generation_trajectory"] = (_corr_entangle * _top5_nbr).fillna(0.0)
    _df["raw_phy_geodesic_jacobi_convergence"] = _geodesic.fillna(0.0)
    _df["raw_phy_geometric_torque_scattering"] = (_top5_nbr * _G_torque).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    _df["raw_phy_interstitial_topological_stress"] = (_top5_brat * (1.0 - _pos) - _top5_srat * _pos).fillna(0.0)
    _df["raw_phy_multiplex_coherence_alignment"] = (_top10_nbr * _score.abs()).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    _df["raw_phy_supernova_core_collapse_vector"] = (-1.0 * (_bot80_nbr - _top10_nbr) * _price_dev).fillna(0.0)

    # === Alpha v15: 生態模擬特徵新增 Raw Columns (20260319) ===
    
    # --- Daily OHLC derived indicators ---
    _open = _df["開盤價"] if "開盤價" in _df.columns else _close
    
    # 1. raw_is_dump_day: 砸盤日 (收盤 < 開盤)
    _df["raw_is_dump_day"] = (_close < _open).astype(float)
    
    # 2. raw_is_founder_day: 創始日 (收盤突破過去10日高點)
    # Use transform to maintain index alignment
    _close_10d_max = _close.groupby(level="StockId").transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).max()
    )
    _df["raw_is_founder_day"] = (_close > _close_10d_max).astype(float)
    
    # 3. raw_is_new_low_5d: 5日新低
    _close_5d_min = _close.groupby(level="StockId").transform(
        lambda x: x.rolling(5, min_periods=1).min()
    )
    _df["raw_is_new_low_5d"] = (_close == _close_5d_min).astype(float)
    
    # 4. raw_is_new_low_20d: 20日新低
    _close_20d_min = _close.groupby(level="StockId").transform(
        lambda x: x.rolling(20, min_periods=1).min()
    )
    _df["raw_is_new_low_20d"] = (_close == _close_20d_min).astype(float)
    
    # --- Broker-level aggregations for new features ---
    
    # 5. raw_buyer_count: 淨買入券商數
    _brk_day["NetBuy"] = _brk_day["BuyQtm"] - _brk_day["SellQtm"]
    _buyer_count = _brk_day[_brk_day["NetBuy"] > 0].groupby(["StockId", "Date"])["BrokerId"].count()
    _df["raw_buyer_count"] = _buyer_count.reindex(_df.index).fillna(0).values
    
    # 6. raw_top10_net_sell_qtm: Top 10 by SellQtm with NetBuy < 0
    _brk_day["SellRank"] = _brk_day.groupby(["StockId", "Date"])["SellQtm"].rank(method="first", ascending=False)
    _top10_sellers = _brk_day[(_brk_day["SellRank"] <= 10) & (_brk_day["NetBuy"] < 0)]
    _top10_net_sell = _top10_sellers.groupby(["StockId", "Date"])["NetBuy"].sum().abs()
    _df["raw_top10_net_sell_qtm"] = _top10_net_sell.reindex(_df.index).fillna(0).values
    
    # 7. raw_top5_sell: Top 5 by SellQtm sum
    _brk_day["SellRankAbs"] = _brk_day.groupby(["StockId", "Date"])["SellQtm"].rank(method="first", ascending=False)
    _top5_sell = _brk_day[_brk_day["SellRankAbs"] <= 5].groupby(["StockId", "Date"])["SellQtm"].sum()
    _df["raw_top5_sell"] = _top5_sell.reindex(_df.index).fillna(0).values
    
    # 8. raw_small_buyers_cnt & raw_large_sellers_cnt: for inverse Alle exhaustion
    _brk_day["SizePct"] = _brk_day.groupby(["StockId", "Date"])["TotalQtm"].rank(pct=True, ascending=True)
    _small_net_buyers = _brk_day[(_brk_day["NetBuy"] > 0) & (_brk_day["SizePct"] <= 0.5)]
    _large_net_sellers = _brk_day[(_brk_day["NetBuy"] < 0) & (_brk_day["SizePct"] >= 0.8)]
    _small_buyers_cnt = _small_net_buyers.groupby(["StockId", "Date"])["BrokerId"].count()
    _large_sellers_cnt = _large_net_sellers.groupby(["StockId", "Date"])["BrokerId"].count()
    _df["raw_small_buyers_cnt"] = _small_buyers_cnt.reindex(_df.index).fillna(0).values
    _df["raw_large_sellers_cnt"] = _large_sellers_cnt.reindex(_df.index).fillna(0).values
    
    # 9. raw_top10_boundary_buy: Top10 買量 at >= daily_high * 0.99
    _df_reset = _df.reset_index()
    _brk_with_price = _brk_raw.copy()
    _df_daily_high = _df_reset[["StockId", "Date"]].copy()
    _df_daily_high["_daily_high"] = _h_d.values
    _brk_with_price = _brk_with_price.merge(_df_daily_high, on=["StockId", "Date"], how="left")
    _brk_day_top10 = _brk_day[_brk_day["RankAbs"] <= 10][["StockId", "Date", "BrokerId"]].drop_duplicates()
    _brk_boundary = _brk_with_price[
        (_brk_with_price["Price"] >= _brk_with_price["_daily_high"] * 0.99) & 
        (_brk_with_price["BuyQtm"] > 0)
    ]
    _brk_boundary_top10 = _brk_boundary.merge(_brk_day_top10, on=["StockId", "Date", "BrokerId"], how="inner")
    _top10_boundary_buy = _brk_boundary_top10.groupby(["StockId", "Date"])["BuyQtm"].sum()
    _df["raw_top10_boundary_buy"] = _top10_boundary_buy.reindex(_df.index).fillna(0).values
    
    # 10. raw_top10_baseline_buy: Top10 買量 at <= daily_low * 1.01
    _df_daily_low = _df_reset[["StockId", "Date"]].copy()
    _df_daily_low["_daily_low"] = _l_d.values
    _brk_baseline = _brk_raw.copy()
    _brk_baseline = _brk_baseline.merge(_df_daily_low, on=["StockId", "Date"], how="left")
    _brk_baseline = _brk_baseline.merge(_brk_day_top10, on=["StockId", "Date", "BrokerId"], how="inner")
    _brk_baseline = _brk_baseline[(_brk_baseline["Price"] <= _brk_baseline["_daily_low"] * 1.01) & (_brk_baseline["BuyQtm"] > 0)]
    _top10_baseline_buy = _brk_baseline.groupby(["StockId", "Date"])["BuyQtm"].sum()
    _df["raw_top10_baseline_buy"] = _top10_baseline_buy.reindex(_df.index).fillna(0).values
    
    # 11. raw_top10_total_buy: Top10 總買量
    _top10_total_buy = _brk_day[_brk_day["RankAbs"] <= 10].groupby(["StockId", "Date"])["BuyQtm"].sum()
    _df["raw_top10_total_buy"] = _top10_total_buy.reindex(_df.index).fillna(0).values
    
    # Alpha v17 GT: raw_top10_total_vol & raw_total_broker_vol (for f_gt_echo_chamber_assortative)
    # Reference computes: top10_ratio = top10_vol / total_all where both are from TotalVol
    _top10_total_vol = _brk_day[_brk_day["RankAbs"] <= 10].groupby(["StockId", "Date"])["TotalQtm"].sum()
    _df["raw_top10_total_vol"] = _top10_total_vol.reindex(_df.index).fillna(0).values
    _total_broker_vol = _brk_day.groupby(["StockId", "Date"])["TotalQtm"].sum()
    _df["raw_total_broker_vol"] = _total_broker_vol.reindex(_df.index).fillna(0).values
    
    # 12. raw_top5_net_buy: Top5 淨買 (needed for f_founder_effect_decay)
    # OLD approach: rank by TotalQtm, then sum NetBuy of top 5
    _df["raw_top5_net_buy"] = _top5_net.reindex(_df.index).fillna(0).values
    
    # Alpha v17: raw_top5_net_buy_nlargest - Top5 by NetBuy directly (for f_gt_costly_waiting_attrition)
    _brk_day["NetBuy"] = _brk_day["BuyQtm"] - _brk_day["SellQtm"]
    
    def _top5_netbuy_nlargest(group):
        return group.nlargest(5, 'NetBuy')['NetBuy'].sum()
    
    _top5_netbuy_nlargest_series = (
        _brk_day.groupby(["StockId", "Date"])
        .apply(_top5_netbuy_nlargest)
    )
    _df["raw_top5_net_buy_nlargest"] = _top5_netbuy_nlargest_series.reindex(_df.index).fillna(0).values
    
    # === Alpha v17 Game Theory: raw_top5_net_sell_qtm ===
    # Top 5 brokers by NetSell (SellQtm - BuyQtm) — for f_gt_second_order_belief_fading
    _brk_day["NetSell"] = _brk_day["SellQtm"] - _brk_day["BuyQtm"]
    _netsell_rank = _brk_day.groupby(["StockId", "Date"])["NetSell"].rank(method="first", ascending=False)
    _top5_netsell = _brk_day[_netsell_rank <= 5].groupby(["StockId", "Date"])["NetSell"].sum()
    _df["raw_top5_net_sell_qtm"] = _top5_netsell.reindex(_df.index).fillna(0).values
    # === End Alpha v17 ===
    
    # === Alpha v17 GT F31-F34: Rolling Top-N Broker Aggregations ===
    # Sort broker-day data for rolling calculations
    _brk_day = _brk_day.sort_values(["StockId", "Date", "BrokerId"])
    
    # 20-day rolling cum_net per broker (shift to prevent leakage)
    _brk_day["roll_net_buy_20d"] = _brk_day.groupby(["StockId", "BrokerId"])["NetBuy"].transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=5).sum()
    )
    
    # Rank brokers by 20-day rolling cum_net within each stock-date
    _brk_day["rank_20d"] = _brk_day.groupby(["StockId", "Date"])["roll_net_buy_20d"].rank(
        method="first", ascending=False
    )
    
    # Filter top 5 and aggregate
    _top5_20d = _brk_day[_brk_day["rank_20d"] <= 5]
    _top5_20d_agg = _top5_20d.groupby(["StockId", "Date"]).agg(
        top5_20d_netbuy=("NetBuy", "sum"),
        top5_20d_positive_days=("NetBuy", lambda x: (x > 0).sum())
    )
    
    # 5-day rolling cum_net per broker
    _brk_day["roll_net_buy_5d"] = _brk_day.groupby(["StockId", "BrokerId"])["NetBuy"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).sum()
    )
    
    # Rank brokers by 5-day rolling cum_net
    _brk_day["rank_5d"] = _brk_day.groupby(["StockId", "Date"])["roll_net_buy_5d"].rank(
        method="first", ascending=False
    )
    
    # Top 5 by 5-day rolling - aggregate NetBuy
    _top5_5d = _brk_day[_brk_day["rank_5d"] <= 5]
    _top5_5d_agg = _top5_5d.groupby(["StockId", "Date"])["NetBuy"].sum().rename("top5_5d_netbuy")
    
    # 10-day rolling cum_net per broker
    _brk_day["roll_net_buy_10d"] = _brk_day.groupby(["StockId", "BrokerId"])["NetBuy"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).sum()
    )
    
    # Rank brokers by 10-day rolling cum_net
    _brk_day["rank_10d"] = _brk_day.groupby(["StockId", "Date"])["roll_net_buy_10d"].rank(
        method="first", ascending=False
    )
    
    # Top 1 by 10-day rolling - aggregate NetBuy and BuyQtm
    _top1_10d = _brk_day[_brk_day["rank_10d"] == 1]
    _top1_10d_agg = _top1_10d.groupby(["StockId", "Date"]).agg(
        top1_10d_netbuy=("NetBuy", "sum"),
        top1_10d_incumbent_buy=("BuyQtm", "sum")
    )
    
    # Merge rolling top-N aggregations to daily df
    _df["top5_20d_netbuy"] = _top5_20d_agg["top5_20d_netbuy"].reindex(_df.index).fillna(0).values
    _df["top5_20d_positive_days"] = _top5_20d_agg["top5_20d_positive_days"].reindex(_df.index).fillna(0).values
    _df["top5_5d_netbuy"] = _top5_5d_agg.reindex(_df.index).fillna(0).values
    _df["top1_10d_netbuy"] = _top1_10d_agg["top1_10d_netbuy"].reindex(_df.index).fillna(0).values
    _df["top1_10d_incumbent_buy"] = _top1_10d_agg["top1_10d_incumbent_buy"].reindex(_df.index).fillna(0).values
    
    # === Price Stability & Shock Detection for F31, F32 ===
    # past_success: 5-day return > 5% (shift to prevent leakage)
    _ret_5d = _close.pct_change(5).shift(1)
    _df["past_success"] = (_ret_5d > 0.05).astype(float)
    
    # today_shock: open < 0.98 * prev_close (shift to prevent leakage)
    _prev_close = _close.shift(1)
    _df["today_shock"] = (_open < 0.98 * _prev_close).astype(float)
    
    # price_stability: 3-day return > -1% (shift to prevent leakage)
    _ret_3d = _close.pct_change(3).shift(1)
    _df["price_stability"] = (_ret_3d > -0.01).astype(float)
    
    # === Volume Surge for F33 ===
    # vol_surge: volume > 1.5 * 20-day rolling mean volume (shift to prevent leakage)
    _vol_20d_mean = _df["成交量(千股)"].groupby(level="StockId").transform(
        lambda x: x.shift(1).rolling(window=20, min_periods=5).mean()
    )
    _df["vol_surge"] = (_df["成交量(千股)"] > 1.5 * _vol_20d_mean).astype(float)
    
    # === End Alpha v17 GT F31-F34 ===
    
    # === End Alpha v15 ===
    
    result = _df.reset_index()
    if return_broker_day:
        return result, broker_day_original
    return result
