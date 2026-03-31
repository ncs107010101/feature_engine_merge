import pandas as pd
import numpy as np

def __zscore_rolling_local(x: pd.Series, window: int) -> pd.Series:
    r = x.rolling(window, min_periods=max(window // 2, 5))
    return (x - r.mean()) / (r.std() + 1e-9)

def preprocess_single_stock_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Strict validation of required columns
    req_cols = [
        "StockId", "收盤價", "最高價", "最低價", "開盤價", 
        "成交量(千股)", "流通在外股數(千股)", "當沖買賣占比"
    ]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"single_stock_daily daily_data is missing required columns: {missing}")

    g = df.groupby("StockId", group_keys=False)

    df["_vol_ma20"] = g["成交量(千股)"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["_vol_ma10"] = g["成交量(千股)"].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df["_prev_close"] = g["收盤價"].shift(1)
    df["_ret_1d"] = g["收盤價"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["_turnover_rate"] = df["成交量(千股)"] / (df["流通在外股數(千股)"] + 1e-9)
    df["_dt_pct"] = df["當沖買賣占比"].fillna(0) / 100.0

    df["_vol_ma20_orig"] = g["成交量(千股)"].transform(lambda x: x.rolling(20).mean())
    df["_vol_ma60"] = g["成交量(千股)"].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df["_short_bal"] = g["融券餘額(千股)"].ffill()
    df["_intra_ret"] = (df["收盤價"] - df["開盤價"]) / (df["開盤價"] + 1e-9)

    # Calculate True Range and ATR
    tr1 = df["最高價"] - df["最低價"]
    tr2 = (df["最高價"] - df["_prev_close"]).abs()
    tr3 = (df["最低價"] - df["_prev_close"]).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df["_atr_5"] = tr.groupby(df["StockId"]).transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["_atr_20"] = tr.groupby(df["StockId"]).transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["_atr_60"] = tr.groupby(df["StockId"]).transform(lambda x: x.rolling(60, min_periods=1).mean())

    # Shared z-scores for composite
    df["_tr_z"] = g["_turnover_rate"].transform(lambda x: __zscore_rolling_local(x, 10))
    df["_dt_z"] = g["_dt_pct"].transform(lambda x: __zscore_rolling_local(x, 10))

    # === GT Alpha v17 Preprocessing: Game Theory Features ===
    # Columns for f_gt_level_k_retail_harvesting, f_gt_reputation_cost_signaling
    df["_ret_5d"] = g["收盤價"].pct_change(5)
    # === F35: f_be_asymmetric_confidence_spillover ===
    # 3日報酬率
    df["_ret_3d"] = g["收盤價"].pct_change(3)
    # 當沖比例20日滾動均值
    df["_dt_mean_20"] = g["_dt_pct"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    # === END F35 ===
    df["_inst_net"] = (
        df["外資買賣超張數"].fillna(0) +
        df["投信買賣超張數"].fillna(0) +
        df["自營買賣超張數(自行)"].fillna(0) +
        df["自營買賣超張數(避險)"].fillna(0)
    )
    df["_inst_net_change"] = df.groupby("StockId")["_inst_net"].diff(1)

    # Columns for f_gt_delayed_disclosure_moral_hazard
    df["_short_total"] = df["融券餘額(千股)"].ffill().fillna(0) + df["借券賣出餘額(千股)"].ffill().fillna(0)
    df["_short_change_5d"] = df.groupby("StockId")["_short_total"].diff(5)
    df["_vol_20d"] = df.groupby("StockId")["_ret_1d"].transform(lambda x: x.rolling(20, min_periods=10).std())

    # Columns for f_gt_strategic_ignorance_demand, f_gt_moral_hazard_leverage
    df["_sma_20"] = df.groupby("StockId")["收盤價"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["_margin_bal"] = df["融資餘額(千股)"].ffill().fillna(0)
    df["_margin_change_3"] = df.groupby("StockId")["_margin_bal"].pct_change(3)
    df["_margin_usage"] = df["_margin_bal"] / (df["流通在外股數(千股)"].fillna(0) + 1e-8)

    # Columns for f_gt_moral_hazard_leverage
    df["_large_holder_pct"] = df["超過1000張集保占比_最近1次發布資料"].ffill().fillna(50) / 100.0

    # Columns for f_gt_unraveling_skepticism_failure
    df["_rev_yoy"] = (
        (df["最近1期單月營收(千元)"].ffill() - df["最近13期單月營收(千元)"].ffill()) /
        (df["最近13期單月營收(千元)"].ffill().abs() + 1e-8)
    )
    df["_retail_count"] = df["未滿400張集保人數_最近1次發布資料"].ffill()
    # === END GT Alpha v17 Preprocessing ===

    # === PHY Alpha v1: f_phy_bic_radiationless_momentum ===
    # BIC_Strength = DailyVolRatio / (HighLowRet + 1e-4)
    # Output = BIC_Strength * IntraRet  (price-scale invariant: all ratios)
    _open_safe = df["開盤價"].replace(0, 1)
    _hl_ret = (df["最高價"] - df["最低價"]) / (_open_safe + 1e-6)
    _vol_20ma_bic = g["成交量(千股)"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    ).replace({0: 1})
    _daily_vol_ratio = df["成交量(千股)"] / _vol_20ma_bic
    _bic_strength = _daily_vol_ratio / (_hl_ret + 1e-4)
    _intra_ret_bic = (df["收盤價"] - df["開盤價"]) / (_open_safe + 1e-6)
    df["raw_phy_bic_radiationless_momentum"] = (
        (_bic_strength * _intra_ret_bic)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    # === END PHY Alpha v1 ===

    return df
