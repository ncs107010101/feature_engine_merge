import pandas as pd
import numpy as np
from .single_stock_broker import preprocess_single_stock_broker
from .single_stock_tick import preprocess_single_stock_tick


def preprocess_cross_tick_broker_daily(data: pd.DataFrame, return_broker_day: bool = False, **kwargs) -> pd.DataFrame:
    if "daily_data" not in kwargs:
        raise ValueError(
            "cross_tick_broker_daily requires daily OHLC data passed as "
            "`daily_data` in kwargs."
        )

    df_daily = kwargs["daily_data"].copy()
    
    req_daily = ["StockId", "Date", "收盤價", "最高價", "最低價", "成交量(千股)", "報酬率", "開盤價"]
    missing = [c for c in req_daily if c not in df_daily.columns]
    if missing:
        raise ValueError(
            f"daily_data is missing required columns: {missing}. "
            "Note: cross_tick_broker_daily features require '收盤價', '報酬率', and '成交量(千股)'."
        )

    broker_daily, broker_day_original = preprocess_single_stock_broker(data, return_broker_day=True)

    df_daily["Date"] = df_daily["Date"].astype(int)
    broker_daily["Date"] = broker_daily["Date"].astype(int)

    merge_cols = ["StockId", "Date"]
    daily_cols_needed = ["收盤價", "最高價", "最低價", "成交量(千股)", "報酬率", "開盤價"]
    if "成交金額(元)" in df_daily.columns:
        daily_cols_needed.append("成交金額(元)")

    _df = df_daily[merge_cols + daily_cols_needed].copy()
    _df = _df.sort_values(["StockId", "Date"])

    broker_daily_for_merge = broker_daily.drop(columns=[c for c in broker_daily.columns if c in daily_cols_needed and c not in ["DailyVWAP"]], errors="ignore")
    _df = _df.merge(broker_daily_for_merge, on=merge_cols, how="left")

    tick_data = kwargs.get("_tick_raw", None)
    if tick_data is not None:
        tick_intermediates = preprocess_single_stock_tick(tick_data)
        tick_intermediates["raw_active_sell"] = tick_intermediates["sell_vol"]
        tick_intermediates["raw_total_active_buy"] = tick_intermediates["buy_vol"]
        
        tick_intermediates["Date"] = tick_intermediates["Date"].astype(int)
        
        tick_data_int = tick_data.copy()
        tick_data_int["Date"] = tick_data_int["Date"].astype(int)
        tick_data_int = tick_data_int.sort_values(["StockId", "Date", "TotalQty"]).reset_index(drop=True)
        small_buy_mask = (tick_data_int["DealCount"] == 1) & (tick_data_int["PrFlag"] == 1)
        small_buy_daily = tick_data_int[small_buy_mask].groupby(["StockId", "Date"]).size().reset_index()
        small_buy_daily.columns = ["StockId", "Date", "raw_small_buy_ticks"]
        
        _df = _df.merge(small_buy_daily, on=merge_cols, how="left")
        if "raw_small_buy_ticks" in _df.columns:
            _df["raw_small_buy_ticks"] = _df["raw_small_buy_ticks"].fillna(0)
        
        tick_daily_cols = ["vwap", "buy_vol", "sell_vol", "total_vol"]
        
        tick_cols_available = [c for c in tick_daily_cols if c in tick_intermediates.columns]
        if tick_cols_available:
            tick_daily = tick_intermediates[merge_cols + tick_cols_available].copy()
            _df = _df.merge(tick_daily, on=merge_cols, how="left")
            for col in tick_cols_available:
                if col in _df.columns:
                    _df[col] = _df[col].fillna(0)

    _df = _df.sort_values(["StockId", "Date"])
    _df = _df.set_index(["StockId", "Date"])

    _daily_vol_shares = _df["成交量(千股)"] * 1000
    _daily_vol_safe   = np.where(_daily_vol_shares == 0, 1.0, _daily_vol_shares)

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

    _top10_nbr = (_top10_net * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe
    _bot80_nbr = (_bot80_net * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe
    _top5_nbr  = (_top5_net  * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe
    _top5_brat = (_top5_buy  * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe
    _top5_srat = (_top5_sell * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe

    _df["raw_top10_net_buy"] = _top10_nbr
    _df["raw_bot80_net_buy"] = _bot80_nbr
    _df["raw_top5_net_buy"] = _top5_nbr
    _df["raw_top5_buy"] = _top5_brat
    _df["raw_top5_sell"] = _top5_srat

    _new_comer = _brk_day[_brk_day["TotalQtm"] <= 5]
    _new_comer_net = _new_comer.groupby(["StockId", "Date"]).apply(
        lambda x: (x["BuyQtm"] - x["SellQtm"]).sum()
    )
    _df["_newcomer_net_momentum"] = (_new_comer_net * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe

    _retail = _brk_day[_brk_day["RankPct"] >= 0.8]
    _retail_net = _retail.groupby(["StockId", "Date"]).apply(
        lambda x: (x["BuyQtm"] - x["SellQtm"]).sum()
    )
    _df["raw_retail_part"] = (_retail_net * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe

    _brk_hhi = _brk_day.groupby(["StockId", "Date"])["TotalQtm"].apply(
        lambda x: (x / x.sum() ** 2).sum()
    )
    _df["raw_broker_hhi"] = _brk_hhi.reindex(_df.index).fillna(0).values

    _brk_day["NetBuy"] = _brk_day["BuyQtm"] - _brk_day["SellQtm"]
    _top5_netbuy_nlargest = _brk_day[_brk_day["RankAbs"] <= 5].groupby(["StockId", "Date"])["NetBuy"].nlargest(3).groupby(["StockId", "Date"]).mean()
    _df["raw_top5_net_buy_nlargest"] = (_top5_netbuy_nlargest * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe

    _brk_day["NetSell"] = _brk_day["SellQtm"] - _brk_day["BuyQtm"]
    _netsell_rank = _brk_day.groupby(["StockId", "Date"])["NetSell"].rank(method="first", ascending=False)
    _top5_netsell = _brk_day[_netsell_rank <= 5].groupby(["StockId", "Date"])["NetSell"].sum()
    _df["raw_top5_net_sell_qtm"] = (_top5_netsell * 1000).reindex(_df.index).fillna(0.0) / _daily_vol_safe

    _close = _df["收盤價"]
    _ret   = _df["報酬率"]
    _h_d   = _df["最高價"]
    _l_d   = _df["最低價"]
    _o_d   = _df["開盤價"]

    _df["_atr_20"] = (
        pd.concat([_h_d - _l_d, (_h_d - _close.shift(1)).abs(), (_l_d - _close.shift(1)).abs()], axis=1)
        .max(axis=1)
        .groupby(_df.index.get_level_values("StockId"))
        .rolling(window=20, min_periods=1)
        .mean()
        .droplevel(0)
    )

    _df["_daily_return"] = _ret

    _df = _df.reset_index()
    
    result = _df
    if return_broker_day:
        return result, broker_day_original
    return result
