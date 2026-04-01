"""
feature_engine.utils
=================================
Shared helper functions for feature calculations.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Z-score and Rolling Transforms
# ---------------------------------------------------------------------------

def zscore_rolling(series: pd.Series, window: int = 20, min_periods: int = None, eps: float = 1e-8) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / (rolling_std + eps)."""
    if min_periods is None:
        min_periods = max(1, window // 2)
    mu = series.rolling(window, min_periods=min_periods).mean()
    sigma = series.rolling(window, min_periods=min_periods).std()
    return (series - mu) / (sigma + eps)


def zscore_transform(series: pd.Series, window: int = 20) -> pd.Series:
    rm = series.rolling(window, min_periods=window).mean()
    rs = series.rolling(window, min_periods=window).std()
    z = ((series - rm) / rs.replace(0, np.nan)).fillna(0)
    return z.clip(-3, 3)


def ewm_smooth(series: pd.Series, span: int = 5, min_periods: int = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(span // 2, 2)
    return series.ewm(span=span, min_periods=min_periods).mean()


def ewm_zscore(series: pd.Series, span: int = 20) -> pd.Series:
    em = series.ewm(span=span, adjust=False).mean()
    es = series.ewm(span=span, adjust=False).std()
    return ((series - em) / es.replace(0, np.nan)).fillna(0)


def ewm_then_zscore(series: pd.Series, ewm_span: int = 5, z_window: int = 20) -> pd.Series:
    """First apply EWM smoothing, then rolling z-score normalization."""
    smoothed = ewm_smooth(series, span=ewm_span)
    return zscore_rolling(smoothed, window=z_window)


# ---------------------------------------------------------------------------
# Time-series Rank and Z-score
# ---------------------------------------------------------------------------

def ts_rank(series: pd.Series, window: int = 20) -> pd.Series:
    try:
        r = series.rolling(window, min_periods=max(1, window // 2)).rank(pct=True)
    except AttributeError:
        r = series.rolling(window, min_periods=max(1, window // 2)).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    return r

def ts_rank_center(series: pd.Series, window: int = 120, min_periods: int = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(window // 2, 5)
    try:
        r = series.rolling(window, min_periods=min_periods).rank(pct=True)
    except AttributeError:
        r = series.rolling(window, min_periods=min_periods).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
    return r - 0.5


def ts_zscore_tick(s: pd.Series, window: int = 20) -> pd.Series:
    r = s.rolling(window, min_periods=max(window // 2, 5))
    return (s - r.mean()) / (r.std() + 1e-9)


def ts_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    return zscore_rolling(series, window)


def mad_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_median = series.rolling(window, min_periods=5).median()
    rolling_mad = series.rolling(window, min_periods=5).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    return (series - rolling_median) / (rolling_mad + 1e-8)


def rolling_rank_pct(series: pd.Series, window: int = 20, min_periods: int = 5) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )


def rolling_slope(series: pd.Series, window: int = 20, min_periods: int = None) -> pd.Series:
    """Rolling linear regression slope using vectorized shift trick."""
    if min_periods is None:
        min_periods = window
    w = window
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    weights = (x - x_mean) / ss_xx
    result = sum(weights[w - 1 - j] * series.shift(j) for j in range(w))
    valid_count = series.notna().rolling(w, min_periods=1).sum()
    result[valid_count < min_periods] = np.nan
    return result

# ---------------------------------------------------------------------------
# Safe Calculations
# ---------------------------------------------------------------------------

def safe_calc(func, *args, default=0.0, **kwargs):
    try:
        result = func(*args, **kwargs)
        if isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
            return default
        return result
    except Exception:
        return default

def safe_skew(x):
    if len(x) < 3:
        return 0.0
    try:
        s = x.skew()
        return 0.0 if pd.isna(s) else s
    except Exception:
        return 0.0


def calc_slope(y):
    if len(y) > 2:
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0] / (y.mean() + 1e-8)
    return 0.0


def safe_div(a, b, default=0.0, fill_eps=1e-8):
    return a / (b + fill_eps)

def safe_calc_func(val):
    if pd.isna(val) or np.isinf(val):
        return 0.0
    return float(val)

# ---------------------------------------------------------------------------
# Feature Post-processing
# ---------------------------------------------------------------------------

def clean_features(
    df: pd.DataFrame,
    feature_cols: list,
    clip_range: tuple = (-5, 5),
    fill_value: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if col in out.columns:
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].fillna(fill_value)
            if clip_range is not None:
                out[col] = out[col].clip(*clip_range)
    return out


def ensure_unique_key(df: pd.DataFrame, key_cols: list = None) -> pd.DataFrame:
    if key_cols is None:
        key_cols = ["StockId", "Date"]
    return df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Information Theory Functions (for IT features)
# ---------------------------------------------------------------------------

EPS = 1e-9
N_BINS = 5
NET_BINS = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]


def binary_entropy(p: float) -> float:
    """Binary entropy H(p) in bits."""
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def empirical_entropy(dist: np.ndarray) -> float:
    """Empirical entropy (nats) from counts or probabilities with Laplace smoothing."""
    d = np.array(dist, dtype=float)
    d = d + EPS
    d = d / d.sum()
    return float(-np.sum(d * np.log(d)))


def empirical_mi(p_x: np.ndarray, p_y: np.ndarray, p_xy: np.ndarray) -> float:
    """
    Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
    p_x: Marginal distribution (length M)
    p_y: Marginal distribution (length N)
    p_xy: Joint distribution (M x N), already smoothed
    """
    p_x = np.array(p_x, dtype=float) + EPS
    p_y = np.array(p_y, dtype=float) + EPS
    p_xy = np.array(p_xy, dtype=float) + EPS
    p_x /= p_x.sum()
    p_y /= p_y.sum()
    p_xy /= p_xy.sum()
    hx = -np.sum(p_x * np.log(p_x))
    hy = -np.sum(p_y * np.log(p_y))
    hxy = -np.sum(p_xy * np.log(p_xy))
    return float(max(0.0, hx + hy - hxy))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) in nats."""
    p = np.array(p, dtype=float) + EPS
    q = np.array(q, dtype=float) + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def phi_coefficient(b_seq: np.ndarray, y_seq: np.ndarray) -> float:
    """Phi Coefficient, range [-1, +1]."""
    b = np.array(b_seq, dtype=int)
    y = np.array(y_seq, dtype=int)
    n11 = np.sum((b == 1) & (y == 1))
    n10 = np.sum((b == 1) & (y == 0))
    n01 = np.sum((b == 0) & (y == 1))
    n00 = np.sum((b == 0) & (y == 0))
    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom < EPS:
        return 0.0
    return float((n11 * n00 - n10 * n01) / denom)


def directed_info_1markov(b_seq: np.ndarray, y_seq: np.ndarray) -> float:
    """Plug-in 1st order Directed Information I(B->Y) estimate."""
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


def directed_info_split(b_seq: np.ndarray, y_seq: np.ndarray):
    """Compute DI contributions for Y_t=1 (up) and Y_t=0 (down)."""
    b = np.array(b_seq, dtype=int)
    y = np.array(y_seq, dtype=int)
    T = len(b)
    if T < 5:
        return 0.0, 0.0

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

    di_up, di_dn = 0.0, 0.0
    n_up, n_dn = 0, 0
    for t in range(1, T):
        ctx = b[t - 1] * 2 + y[t - 1]
        yt = y[t]
        yt1 = y[t - 1]
        contrib = np.log(p_full[ctx, yt] / p_y1[yt1, yt])
        if yt == 1:
            di_up += contrib
            n_up += 1
        else:
            di_dn += contrib
            n_dn += 1

    di_up = di_up / max(n_up, 1)
    di_dn = di_dn / max(n_dn, 1)
    return float(di_up), float(di_dn)


def np_logloss_gain(b_seq: np.ndarray, y_seq: np.ndarray):
    """Non-parametric log-loss gain: Gain_up - Gain_dn."""
    b = np.array(b_seq[:-1])
    y = np.array(y_seq[1:])
    if len(b) < 5:
        return 0.0

    EPS_p = 0.01 / len(b)

    p_up = (y == 1).mean()
    p_dn = 1 - p_up
    p_up = np.clip(p_up, EPS_p, 1 - EPS_p)

    mask_buy = b == 1
    mask_sell = b == 0

    def cond_prob(mask, target):
        n = mask.sum()
        if n == 0:
            return 0.5
        cnt = ((y[mask] == target)).sum()
        return np.clip((cnt + EPS_p * 5) / (n + EPS_p * 10), EPS_p, 1 - EPS_p)

    p_up_buy = cond_prob(mask_buy, 1)
    p_dn_sell = cond_prob(mask_sell, 0)

    gain_up = np.log(p_up_buy / p_up)
    gain_dn = np.log(p_dn_sell / p_dn)
    return float(gain_up - gain_dn)


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score post-processing to eliminate magnitude differences."""
    mu = series.rolling(window, min_periods=5).mean()
    sigma = series.rolling(window, min_periods=5).std()
    return (series - mu) / (sigma + EPS)


# ---------------------------------------------------------------------------
# Rolling Window Information Theory Helpers (for F12-F15)
# ---------------------------------------------------------------------------

WINDOW = 20
MIN_OBS = 10


def rolling_binary_sequences(df: pd.DataFrame,
                              net_col: str = "raw_big_net_ratio",
                              tick_col: str = "raw_p_active_up") -> pd.DataFrame:
    """
    Create rolling binary sequences B (broker direction) and Y (tick direction).
    B = 1 if net_ratio > 0, else 0
    Y = 1 if p_active_up > 0.5, else 0
    """
    df = df.sort_values(["StockId", "Date"]).copy()
    df["B_seq"] = (df[net_col] > 0).astype(int)
    df["Y_seq"] = (df[tick_col] > 0.5).astype(int)
    return df


def compute_causal_kelly_growth_vectorized(win_b: np.ndarray, win_y: np.ndarray,
                                           b_today: int) -> float:
    """
    因果凱利增量：f* - f_base
    Vectorized per-row calculation for F12.
    """
    b = win_b[:-1]  # B_{t-1}
    y = win_y[1:]   # Y_t
    if len(b) < MIN_OBS:
        return np.nan

    EPS_p = 0.5 / (len(b) + 1)
    p_base = np.clip(y.mean(), EPS_p, 1 - EPS_p)
    f_base = 2 * p_base - 1

    mask_buy = b == 1
    mask_sell = b == 0

    def cond_p(mask, target=1):
        n = mask.sum()
        if n < 2:
            return p_base
        cnt = (y[mask] == target).sum()
        return np.clip((cnt + EPS_p) / (n + EPS_p * 2), EPS_p, 1 - EPS_p)

    p_cond = cond_p(mask_buy, 1) if b_today == 1 else cond_p(mask_sell, 1)
    f_star = 2 * p_cond - 1
    return float(f_star - f_base)


# ---------------------------------------------------------------------------
# Price-Time Probability and Tick Burst Helpers
# ---------------------------------------------------------------------------

def calculate_price_time_probability(
    tick_df: pd.DataFrame,
    time_col: str = "Date",
    price_col: str = "DealPrice",
    cond_mask: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute price-time probability: V_condition(Price) / V_total(Price)
    
    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with Date, DealPrice, DealCount columns
    time_col : str
        Name of date column
    price_col : str
        Name of price column
    cond_mask : pd.Series
        Boolean mask indicating which ticks meet the condition
        
    Returns
    -------
    pd.DataFrame
        DataFrame with [time_col, price_col, 'P_time'] columns
    """
    if cond_mask is None or cond_mask.empty:
        return pd.DataFrame(columns=[time_col, price_col, "P_time"])
    
    if len(tick_df) == 0:
        return pd.DataFrame(columns=[time_col, price_col, "P_time"])
    
    EPS_LOCAL = 1e-9
    
    # Total volume per price
    total_vol = tick_df.groupby([time_col, price_col])["DealCount"].sum().reset_index(name="TotalVol")
    
    # Conditional volume per price
    cond_vol = tick_df[cond_mask].groupby([time_col, price_col])["DealCount"].sum().reset_index(name="CondVol")
    
    prob_df = pd.merge(total_vol, cond_vol, on=[time_col, price_col], how="left")
    prob_df["CondVol"] = prob_df["CondVol"].fillna(0)
    prob_df["P_time"] = prob_df["CondVol"] / (prob_df["TotalVol"] + EPS_LOCAL)
    
    return prob_df[[time_col, price_col, "P_time"]]


def pct_change_zscore(
    series: pd.Series, 
    pct_periods: int = 5, 
    zscore_window: int = 42,
    zscore_min_periods: int = None
) -> pd.Series:
    """
    Compute pct_change followed by rolling z-score.
    Used for features like f_predator_driven_extinction_rate.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    pct_periods : int
        Periods for pct_change
    zscore_window : int
        Window for rolling z-score
    zscore_min_periods : int
        Minimum periods for rolling (default: zscore_window // 2)
        
    Returns
    -------
    pd.Series
        Z-scored percentage changes
    """
    if zscore_min_periods is None:
        zscore_min_periods = max(2, zscore_window // 4)
    
    EPS_LOCAL = 1e-9
    
    # Compute pct_change
    pct = series.pct_change(periods=pct_periods)
    
    # Handle inf values from pct_change
    pct = pct.replace([np.inf, -np.inf], np.nan)
    
    # Rolling z-score
    mu = pct.rolling(zscore_window, min_periods=zscore_min_periods).mean()
    sigma = pct.rolling(zscore_window, min_periods=zscore_min_periods).std(ddof=0)
    
    zscore = (pct - mu) / (sigma + EPS_LOCAL)
    
    return zscore


def identify_burst_minute(
    tick_df: pd.DataFrame,
    window_seconds: int = 60
) -> pd.DataFrame:
    """
    Identify the burst minute (minute/window with max active buy volume).
    
    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with Date, DealTimeSecond, PrFlag, DealCount
    window_seconds : int
        Window size in seconds (default 60 = 1 minute)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Date and window index marked as burst
    """
    if len(tick_df) == 0:
        return pd.DataFrame(columns=["Date", "WindowIdx", "is_burst"])
    
    df = tick_df.copy()
    df["IsActiveBuy"] = (df["PrFlag"] == 1).astype(int)
    df["WindowIdx"] = df["DealTimeSecond"] // window_seconds
    
    # Aggregate by window
    window_vol = df.groupby(["Date", "WindowIdx"]).agg(
        ActiveBuySum=("IsActiveBuy", "sum"),
        TickCount=("PrFlag", "count"),
    ).reset_index()
    
    # Find max active buy window per day
    burst_idx = window_vol.groupby("Date")["ActiveBuySum"].idxmax()
    burst_df = window_vol.loc[burst_idx, ["Date", "WindowIdx"]].copy()
    burst_df["is_burst"] = True
    
    return burst_df


def compute_lautum_penalty_vectorized(win_b: np.ndarray, win_y: np.ndarray,
                                     net_ratio_today: float) -> float:
    """
    Lautum penalty: -D_KL(P(B|Y_lag) || P(B)) * net_ratio_today
    Vectorized per-row calculation for F13.
    """
    b = win_b[1:]   # B_t
    y = win_y[:-1]  # Y_{t-1}
    if len(b) < MIN_OBS:
        return np.nan

    p_b = np.array([(b == 0).mean(), (b == 1).mean()]) + 1e-3
    p_b /= p_b.sum()

    mask_y1 = y == 1
    mask_y0 = y == 0

    def cond_b(mask):
        sub = b[mask]
        if len(sub) < 2:
            return p_b.copy()
        d = np.array([(sub == 0).sum(), (sub == 1).sum()], dtype=float) + 1e-3
        return d / d.sum()

    p_b_given_y1 = cond_b(mask_y1)
    p_b_given_y0 = cond_b(mask_y0)

    p_y1 = mask_y1.mean()
    kl_val = p_y1 * kl_divergence(p_b_given_y1, p_b) + \
             (1 - p_y1) * kl_divergence(p_b_given_y0, p_b)

    return float(-kl_val * net_ratio_today)
