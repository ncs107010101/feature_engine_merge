"""
NLD (Nonlinear Dynamics) Feature Preprocessing Additions
These are the raw daily value calculations for the 16 new NLD features.
To be appended to preprocessing/single_stock_tick.py
"""

# ============================================================================
# === Alpha NLD: 16 New Nonlinear Dynamics Features (Batch from new_feature_code)
# ============================================================================
# All features follow the pattern: compute daily raw value → store in day_agg
# The feature scripts will apply zscore_rolling with eps=1e-10 and safe_clip_fillna

# ---- Helper functions for NLD features ----
def _price_to_three_states(dp: np.ndarray) -> np.ndarray:
    """Convert price changes to discrete 3 states: Up=0, Zero=1, Down=2"""
    states = np.ones(len(dp), dtype=int)  # default: Zero
    states[dp > 0] = 0  # Up
    states[dp < 0] = 2  # Down
    return states

def _build_transition_matrix(states: np.ndarray, n_states: int, laplace_smooth: float = 1.0) -> np.ndarray:
    """Build Laplace-smoothed transition matrix"""
    matrix = np.full((n_states, n_states), laplace_smooth)
    for i in range(len(states) - 1):
        s_from = int(states[i])
        s_to = int(states[i + 1])
        if 0 <= s_from < n_states and 0 <= s_to < n_states:
            matrix[s_from, s_to] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return matrix / row_sums

def _stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Compute stationary distribution (left eigenvector)"""
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

def _leading_eigenvalue_real(matrix: np.ndarray) -> float:
    """Extract dominant eigenvalue (real part)"""
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        idx = np.argmax(np.abs(eigenvalues))
        return eigenvalues[idx].real
    except Exception:
        return 0.0

def _eigenvalue_gap(matrix: np.ndarray) -> float:
    """Compute eigenvalue gap |λ1| - |λ2|"""
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        abs_ev = np.sort(np.abs(eigenvalues.real))[::-1]
        if len(abs_ev) >= 2:
            return abs_ev[0] - abs_ev[1]
        return 0.0
    except Exception:
        return 0.0

# ---- 1. f_nld_bifurcation_delay_eigen ----
# 15-min bin (ΔP, ΔV) covariance → eigenvalue gap → min gap acceleration × asym_sign
def _bifurcation_delay(day_df):
    prices = day_df['DealPrice'].values
    volumes = day_df['DealCount'].values.astype(float)
    time_sec = day_df['DealTimeSecond'].values
    if len(prices) < 30:
        return np.nan
    dp = np.diff(prices)
    dv = np.diff(volumes)
    t = time_sec[1:]
    bin_size = 900  # 15 minutes
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
# 5-min bin KDE bimodal → 2×2 transition matrix → eigenvector · acceleration vector
# Note: Uses find_bimodal_peaks from utils - we'll use a simplified version here
def _bimodal_eigen_proj(day_df):
    from scipy.ndimage import gaussian_filter1d
    from scipy import signal
    prices = day_df['DealPrice'].values
    time_sec = day_df['DealTimeSecond'].values
    if len(prices) < 100:
        return np.nan
    bin_size = 300  # 5 minutes
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
        # Find bimodal peaks using histogram
        try:
            hist, bin_edges = np.histogram(p_bin, bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
            peaks_idx = signal.argrelextrema(smoothed, np.greater, order=2)[0]
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
# 100-tick bin AC(1) → arctan2(Vol, AC1) → d²φ/dt² × sign(OFI)
def _critical_slowing(day_df):
    prices = day_df['DealPrice'].values
    volumes = day_df['DealCount'].values.astype(float)
    prflag = day_df['PrFlag'].values if 'PrFlag' in day_df.columns else None
    if len(prices) < 100:
        return np.nan
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
# 3D (ΔP, d(ΔP)/dt, d²(ΔP)/dt²) → cross product Z × sign(ΔVol)
def _dtb_singularity(day_df):
    prices = day_df['DealPrice'].values
    volumes = day_df['DealCount'].values.astype(float)
    if len(prices) < 30:
        return np.nan
    dp = np.diff(prices)
    dp_s = pd.Series(dp).ewm(span=5, min_periods=1).mean().values
    ddp = np.diff(dp_s)
    n = min(len(dp) - 1, len(ddp))
    if n < 10:
        return 0.0
    x = dp[1:n+1]
    y = dp_s[1:n+1]
    z = ddp[:n]
    # Cross product Z component: x1*y2 - x2*y1 for XY, but we need full 3D
    # v1 × v2: (y1*z2 - z1*y2, z1*x2 - x1*z2, x1*y2 - y1*x2)
    # We use the norm of cross product as scalar
    cx = y[:-1]*z[1:] - z[:-1]*y[1:]
    cy = z[:-1]*x[1:] - x[:-1]*z[1:]
    cz = x[:-1]*y[1:] - y[:-1]*x[1:]
    cross_norm = np.sqrt(cx**2 + cy**2 + cz**2)
    dv = np.diff(volumes)
    dv_sign = np.sign(dv[2:n+1])
    min_len = min(len(cross_norm), len(dv_sign))
    if min_len == 0:
        return 0.0
    weighted = cross_norm[:min_len] * dv_sign[:min_len]
    return np.mean(weighted)

day_agg["raw_nld_dtb_singularity_cross"] = df.groupby(["StockId", "Date_int"]).apply(_dtb_singularity).reindex(day_agg.index).values

# ---- 5. f_nld_hopf_eigen_asymmetry ----
# ΔP 3-state → 3×3 transition matrix → max_eigen × (P(U→U) - P(D→D))
def _hopf_eigen(day_df):
    prices = day_df['DealPrice'].values
    if len(prices) < 20:
        return np.nan
    dp = np.diff(prices)
    states = _price_to_three_states(dp)
    tm = _build_transition_matrix(states, n_states=3, laplace_smooth=1.0)
    max_ev = _leading_eigenvalue_real(tm)
    asym = tm[0, 0] - tm[2, 2]  # P(Up→Up) - P(Down→Down)
    return max_ev * asym

day_agg["raw_nld_hopf_eigen_asymmetry"] = df.groupby(["StockId", "Date_int"]).apply(_hopf_eigen).reindex(day_agg.index).values

# ---- 6. f_nld_hysteresis_area ----
# 50-tick bin (ΔP, ΔVol) trajectory signed area
def _hysteresis_area(day_df):
    prices = day_df['DealPrice'].values
    volumes = day_df['DealCount'].values.astype(float)
    if len(prices) < 50:
        return np.nan
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
# tick-to-tick 3-state → stationary distribution · [1,0,-1]
def _limit_cycle_collapse(day_df):
    prices = day_df['DealPrice'].values
    if len(prices) < 20:
        return np.nan
    dp = np.diff(prices)
    states = _price_to_three_states(dp)
    tm = _build_transition_matrix(states, n_states=3, laplace_smooth=1.0)
    pi = _stationary_distribution(tm)
    projection = np.array([1.0, 0.0, -1.0])
    return np.dot(pi, projection)

day_agg["raw_nld_limit_cycle_collapse_markov"] = df.groupby(["StockId", "Date_int"]).apply(_limit_cycle_collapse).reindex(day_agg.index).values

# ---- 8. f_nld_limit_cycle_radius_accel ----
# R=√(ΔP²+ΔV²) → EWM → d²R/dt² × sign(close-open)
def _limit_cycle_radius(day_df):
    prices = day_df['DealPrice'].values
    volumes = day_df['DealCount'].values.astype(float)
    if len(prices) < 20:
        return np.nan
    dp = np.diff(prices)
    dv = np.diff(volumes)
    dp_std = np.std(dp) + 1e-10
    dv_std = np.std(dv) + 1e-10
    dp_norm = dp / dp_std
    dv_norm = dv / dv_std
    R = np.sqrt(dp_norm**2 + dv_norm**2)
    R_series = pd.Series(R)
    # EWM smoothing then second derivative
    R_ewm = R_series.ewm(span=10, min_periods=1).mean()
    d2R = R_ewm.diff().diff()
    mean_d2R = d2R.mean()
    if not np.isfinite(mean_d2R):
        return 0.0
    asym = np.sign(prices[-1] - prices[0])
    return mean_d2R * asym

day_agg["raw_nld_limit_cycle_radius_accel"] = df.groupby(["StockId", "Date_int"]).apply(_limit_cycle_radius).reindex(day_agg.index).values

# ---- 9. f_nld_lyapunov_divergence_accel ----
# phase space (P, Δt) trajectory divergence → d²D/dt² × sign(ΔP)
def _lyapunov_divergence(day_df):
    prices = day_df['DealPrice'].values
    time_sec = day_df['DealTimeSecond'].values.astype(float)
    if len(prices) < 30:
        return np.nan
    p_std = np.std(prices) + 1e-10
    t_std = np.std(np.diff(time_sec)) + 1e-10
    p_norm = prices / p_std
    dt = np.diff(time_sec)
    dt_norm = dt / t_std
    k = 10  # delay embedding
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
# 3D (VWAP, ΔP, ΔVol) → cross product norm → d²/dt² × sign(ΔP)
def _neimark_sacker(day_df):
    prices = day_df['DealPrice'].values
    volumes = day_df['DealCount'].values.astype(float)
    if len(prices) < 30:
        return np.nan
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
# VWAP cross intervals → d² × sign(Price-VWAP)
def _saddle_loop(day_df):
    prices = day_df['DealPrice'].values
    volumes = day_df['DealCount'].values.astype(float)
    if len(prices) < 50:
        return np.nan
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
# arctan2(SellPr-DealPrice, DealPrice-BuyPr) → d²φ/dt² extreme
# Requires BuyPr and SellPr columns
def _saddle_node_phase(day_df):
    if 'BuyPr' not in day_df.columns or 'SellPr' not in day_df.columns:
        return np.nan
    prices = day_df['DealPrice'].values
    buy_pr = day_df['BuyPr'].values
    sell_pr = day_df['SellPr'].values
    if len(prices) < 20:
        return np.nan
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
# KDE bimodal peaks → mode_vector · VWAP_acceleration_vector
def _stochastic_bifurcation(day_df):
    from scipy.ndimage import gaussian_filter1d
    from scipy import signal
    prices = day_df['DealPrice'].values
    if len(prices) < 50:
        return np.nan
    # Find bimodal peaks
    try:
        hist, bin_edges = np.histogram(prices, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
        peaks_idx = signal.argrelextrema(smoothed, np.greater, order=2)[0]
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
# arctan2(ΔP, ΔVol) → EWM → d²θ/dt² × AsymSign
def _stochastic_focusing(day_df):
    prices = day_df['DealPrice'].values
    volumes = day_df['DealCount'].values.astype(float)
    if len(prices) < 30:
        return np.nan
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
# d²P/dt² · ((CeilPr-DealPrice) - (DealPrice-FloorPr)) weighted mean
# Requires CeilPr and FloorPr columns
def _transient_attractor(day_df):
    if 'CeilPr' not in day_df.columns or 'FloorPr' not in day_df.columns:
        return np.nan
    prices = day_df['DealPrice'].values
    ceil_pr = day_df['CeilPr'].values
    floor_pr = day_df['FloorPr'].values
    if len(prices) < 20:
        return np.nan
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

# ============================================================================
# === End Alpha NLD: 16 New Nonlinear Dynamics Features ===
# ============================================================================
