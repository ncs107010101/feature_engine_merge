"""
非線性動力學特徵工程 - 公用算子模組
===================================
提供所有特徵共用的數學算子和工具函數。

使用方式:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common_utils import *
"""
import numpy as np
import pandas as pd
from scipy import signal


# ============================================================
# 1. 二階導數與平滑
# ============================================================

def safe_second_derivative(series: pd.Series, ewm_span: int = 5) -> pd.Series:
    """
    EWM 平滑 → 二階差分。避免直接對高頻噪聲二階導。
    
    Parameters
    ----------
    series : pd.Series
        輸入時間序列（日內 tick 級別）
    ewm_span : int
        EWM 平滑窗口
    
    Returns
    -------
    pd.Series : 二階導數序列
    """
    smoothed = series.ewm(span=ewm_span, min_periods=1).mean()
    return smoothed.diff().diff()


def safe_first_derivative(series: pd.Series, ewm_span: int = 5) -> pd.Series:
    """EWM 平滑 → 一階差分"""
    smoothed = series.ewm(span=ewm_span, min_periods=1).mean()
    return smoothed.diff()


# ============================================================
# 2. 相位角與向量運算
# ============================================================

def phase_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """計算相位角 arctan2(y, x)"""
    return np.arctan2(y, x)


def cross_product_2d(x1, y1, x2, y2):
    """
    2D 外積 (有向面積): x1*y2 - x2*y1
    正值 = 逆時針旋轉, 負值 = 順時針旋轉
    """
    return x1 * y2 - x2 * y1


def cross_product_3d_z(v1_x, v1_y, v1_z, v2_x, v2_y, v2_z):
    """3D 外積的 Z 軸分量: v1_x*v2_y - v1_y*v2_x"""
    return v1_x * v2_y - v1_y * v2_x


def dot_product_2d(x1, y1, x2, y2):
    """2D 向量內積"""
    return x1 * x2 + y1 * y2


# ============================================================
# 3. 轉移矩陣與特徵值
# ============================================================

def build_transition_matrix(states: np.ndarray, n_states: int, 
                            laplace_smooth: float = 1.0) -> np.ndarray:
    """
    Laplace 平滑的轉移矩陣
    
    Parameters
    ----------
    states : np.ndarray
        狀態序列 (整數, 0-indexed)
    n_states : int
        狀態數量
    laplace_smooth : float
        Laplace 平滑參數 (各計數 +laplace_smooth)
    
    Returns
    -------
    np.ndarray : shape (n_states, n_states) 的轉移機率矩陣
    """
    matrix = np.full((n_states, n_states), laplace_smooth)
    for i in range(len(states) - 1):
        s_from = int(states[i])
        s_to = int(states[i + 1])
        if 0 <= s_from < n_states and 0 <= s_to < n_states:
            matrix[s_from, s_to] += 1
    # 正規化為機率
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return matrix / row_sums


def leading_eigenvalue_real(matrix: np.ndarray) -> float:
    """提取主特徵值（最大絕對值），取實部"""
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        idx = np.argmax(np.abs(eigenvalues))
        return eigenvalues[idx].real
    except Exception:
        return 0.0


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """
    計算轉移矩陣的平穩分佈（左特徵向量）
    
    Returns
    -------
    np.ndarray : 平穩分佈向量（和為 1）
    """
    try:
        n = matrix.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        # 找最接近 1 的特徵值
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


def eigenvalue_gap(matrix: np.ndarray) -> float:
    """計算特徵值差距 |λ1| - |λ2|"""
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        abs_ev = np.sort(np.abs(eigenvalues.real))[::-1]
        if len(abs_ev) >= 2:
            return abs_ev[0] - abs_ev[1]
        return 0.0
    except Exception:
        return 0.0


# ============================================================
# 4. 日頻化與標準化
# ============================================================

def rolling_zscore(series: pd.Series, window: int = 20, 
                   min_periods: int = None) -> pd.Series:
    """
    滾動 Z-score 標準化
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)


def asym_sign_series(close: pd.Series, open_: pd.Series) -> pd.Series:
    """不對稱符號: sign(close - open)"""
    return np.sign(close - open_)


def safe_clip_fillna(series: pd.Series, lo: float = -5.0, hi: float = 5.0) -> pd.Series:
    """標準後處理: replace inf → NaN → fillna(0) → clip"""
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lo, hi)


# ============================================================
# 5. KDE 雙峰擬合
# ============================================================

def find_bimodal_peaks(prices: np.ndarray, n_bins: int = 50) -> tuple:
    """
    使用 KDE 尋找雙峰分佈的兩個峰值
    
    Returns
    -------
    (peak_high, peak_low) : tuple of float
        如果找不到雙峰，返回 (mean, mean)
    """
    if len(prices) < 10:
        m = np.mean(prices) if len(prices) > 0 else 0.0
        return (m, m)
    
    try:
        # 使用直方圖近似 KDE（效能更好）
        hist, bin_edges = np.histogram(prices, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 平滑
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
        
        # 找局部極大值
        peaks_idx = signal.argrelextrema(smoothed, np.greater, order=3)[0]
        
        if len(peaks_idx) >= 2:
            # 取最高的兩個峰
            peak_heights = smoothed[peaks_idx]
            top2_idx = peaks_idx[np.argsort(peak_heights)[-2:]]
            top2_idx = np.sort(top2_idx)
            return (bin_centers[top2_idx[1]], bin_centers[top2_idx[0]])
        else:
            m = np.mean(prices)
            return (m, m)
    except Exception:
        m = np.mean(prices)
        return (m, m)


# ============================================================
# 6. 離散狀態編碼
# ============================================================

def price_to_three_states(dp: np.ndarray) -> np.ndarray:
    """
    將價格跳動轉為離散三態
    dp > 0 → 0 (Up)
    dp == 0 → 1 (Zero) 
    dp < 0 → 2 (Down)
    """
    states = np.ones(len(dp), dtype=int)  # default: Zero
    states[dp > 0] = 0  # Up
    states[dp < 0] = 2  # Down
    return states


def prflag_to_ei_states(prflag: np.ndarray) -> np.ndarray:
    """
    PrFlag to E/I states
    PrFlag=1 (外盤/主動買) → 0 (E/Excitatory)
    PrFlag=0 (內盤/主動賣) → 1 (I/Inhibitory)
    PrFlag=2 (其他) → 視為 I
    """
    states = np.ones(len(prflag), dtype=int)  # default: I
    states[prflag == 1] = 0  # E
    return states


def prflag_to_four_states(prflag: np.ndarray) -> np.ndarray:
    """
    連續 PrFlag 轉四態:
    0: 連續內盤 (I→I)
    1: 內轉外 (I→E)
    2: 外轉內 (E→I)
    3: 連續外盤 (E→E)
    """
    ei = prflag_to_ei_states(prflag)
    states = np.zeros(len(ei), dtype=int)
    for i in range(1, len(ei)):
        prev, curr = ei[i-1], ei[i]
        if prev == 1 and curr == 1:
            states[i] = 0  # I→I
        elif prev == 1 and curr == 0:
            states[i] = 1  # I→E
        elif prev == 0 and curr == 1:
            states[i] = 2  # E→I
        elif prev == 0 and curr == 0:
            states[i] = 3  # E→E
    return states


# ============================================================
# 7. 特徵計算統一接口
# ============================================================

def compute_single_stock_feature(df_stock: pd.DataFrame, 
                                  feature_func, 
                                  feature_name: str) -> pd.DataFrame:
    """
    對單一股票計算特徵，包含日頻聚合和跨日標準化。
    
    Parameters
    ----------
    df_stock : pd.DataFrame
        單一股票的 tick 資料（已排序）
    feature_func : callable
        計算原始日頻值的函數，簽名: func(day_ticks) -> float
    feature_name : str
        特徵名稱
        
    Returns
    -------
    pd.DataFrame with [StockId, Date, feature_name]
    """
    stock_id = df_stock['StockId'].iloc[0]
    
    # 按日分組計算
    results = []
    for date, day_group in df_stock.groupby('Date'):
        try:
            val = feature_func(day_group)
        except Exception:
            val = 0.0
        if not np.isfinite(val):
            val = 0.0
        results.append({'StockId': stock_id, 'Date': date, 'raw': val})
    
    result_df = pd.DataFrame(results)
    if len(result_df) == 0:
        return pd.DataFrame(columns=['StockId', 'Date', feature_name])
    
    # 跨日 rolling z-score
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    result_df[feature_name] = rolling_zscore(result_df['raw'], window=20)
    result_df[feature_name] = safe_clip_fillna(result_df[feature_name])
    
    return result_df[['StockId', 'Date', feature_name]]
