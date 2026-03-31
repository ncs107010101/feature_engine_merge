# Feature Engine 開發與編寫規範

本指南旨在為未來接手 `feature_engine` 模組的 LLM (大型語言模型) 及開發人員提供明確的架構說明與程式碼編寫規範。閱讀並嚴格遵守這些準則，能確保該模組具備「極致效能」、「Clean Code」與「100% 獨立解耦」的工業生產級水準。

---

## 總體原則 (Core Principles)

1. **單一職責原則 (Single Responsibility Principle, SRP)**：
   - 每個特徵必須獨立成為一個 Python 檔案 (`calculate_f_{feature_name}.py`)。
   - `preprocessing/` 僅負責「原始資料 (Raw Data)」到「日線中間變數 (Daily Intermediates)」的轉換，**嚴禁**在這裡計算跨日時間序列 (Time-Series) 或 Rank/EWM 等平滑操作。
2. **DRY (Don't Repeat Yourself)**：
   - 「群組化 (Groupby)」與「資料合併 (Merge)」是極其昂貴的操作。多個特徵共用的聚合結果 (例如分點前5大買超、開高低收、VWAP)，必須在 `preprocessing/` 中算好，**嚴禁**在各別的 `calculate_*.py` 中重複執行。
3. **極致向量化 (Extreme Vectorization)**：
   - **盡量避免使用 `.apply(lambda x: ...)` 或 `iterrows()` iterating through dataframe rows**。請善用 Pandas 的 `transform`、`np.where`、`np.sign` 與廣播數學運算 (Broadcasting) 來達成效能最佳化。
   - 僅在無法向量化時才使用 `.apply()` 或迴圈。
4. **組合式資料路由 (Combinatorial Routing)**：
   - 特徵不是依據演算法分類，而是依據 **「它需要什麼原始資料」** 來進行分流。

---

## 1. 資料集組合方式 (Data Combinations)

`feature_engine` 的核心靈魂在於動態路由 (Router)。當使用者呼叫 `compute_features` 時，系統會依據預先定義的資料組合，去撈取那些符合條件的特徵腳本執行。

目前的基礎組合 (基於歷史序列資料)：
- `single_stock_daily`: 僅需日K歷史資料 (`df_daily`)。
- `single_stock_tick`: 需逐筆交易歷史資料 (`df_tick`)。
- `single_stock_broker`: 需分點進出歷史資料 (`df_broker`)。
- `cross_broker_tick`: 同時需要多種異質歷史資料 (如 `df_broker` + `df_tick`) 進行交叉比對。
- `cross_broker_daily`: 券商分點資料 + 日頻 OHLC 交叉比對，用於 PHY 物理現象特徵與 Alpha v15 生態模擬特徵。需透過 `kwargs` 傳入 `daily_data=df_daily`。
- `cross_tick_daily`: 逐筆明細資料 + 日頻 OHLC 交叉比對，用於 Alpha v16 生態模擬特徵與 Alpha v17 遊戲理論特徵。需透過 `kwargs` 傳入 `daily_data=df_daily`。
- `cross_tick_broker_daily`: 券商分點 + 逐筆明細 + 日頻 OHLC 交叉比對，同時需要三種資料。需透過 `kwargs` 傳入 `daily_data=df_daily` 和 `tick_data=df_tick`。

### 📌 新增不同資料集組合 (How to Add a New Combination)
若未來引入了新資料 (例如：財報資料、新聞情感資料)，請依照下列步驟擴充組合：
1. **建立資料夾體系**：
   - 在 `preprocessing/` 新增 `single_stock_financial.py`。
   - 在 `features/` 底下新增目錄 `single_stock_financial/`。
2. **在 `api.py` 註冊路由**：
   - 開放 `compute_features` 接受該新資料的 `**kwargs` 傳遞參數 (例如 `compute_features('single_stock_financial', df_finance)` )。
3. **在 `preprocessing/__init__.py` 註冊入口**：
   - 在 `preprocess` 函式的 `if/elif` 條件樹中，加入新組合的預處理轉發邏輯。

---

## 2. 預處理層編寫規範 (Preprocessing Rules)

預處理層位於 `feature_engine/preprocessing/`，其目的是**「將龐大的 Raw Data 降維至『日線層級 (Date-Level)』的共用變數表」**。

### ✅ DOs:
- 將相同維度 (GroupBy `StockId`, `Date`) 的運算盡可能合併，利用 `agg` 一次性完成。
- 若需進行條件加總 (Filtered Sum，例如：只要早盤 ActiveBuy)，先用 `np.where(mask, value, 0)` 將數值寫入暫存欄位，之後再**與其他指標一起**用一次 `groupby().sum()` 解決，避免重複的 merge 產生龐大的記憶體開銷。
- 檢查來源欄位與對日期 `Date` (`YYYYMMDD` int 型態) 進行轉型與排序。
- **日線時間序列計算必須使用 `groupby().transform(lambda x: x.rolling(...))` 或 `groupby().transform(lambda x: x.shift(...))`** 以維持 Index 對齊。**嚴禁使用 `reset_index(drop=True)` 後再進行時間序列運算**，否則會造成 Index 對齊錯誤。

### ❌ DON'Ts:
- **絕對不允許**在預處理層直接算出最終要輸出的 Time-Series 特徵 (如 Rolling Z-Score、EWM、Rank)。
- **避免使用迴圈與 `.apply()`**，除非是計算無法輕易繞開的複雜自訂分配 (例如極端自訂偏度)。
- **嚴禁使用 `reset_index(level="StockId", drop=True)` 後再進行跨 Series 比較或運算**，這會破壞 Index 對齊導致 "Can only compare identically-labeled Series objects" 錯誤。

---

## 3. 特徵腳本編寫規範 (Feature Script Rules)

所有的特徵腳本位於 `feature_engine/features/{combination_name}/` 之下。

### 檔案與類別命名規範：
- 檔名必須統一為：`calculate_{你實際的特徵名稱}.py`。 (例如 `calculate_f_morning_smart_buy_ewm3.py`)
- 不可在特徵名中使用特殊符號 (如 `%`)，請寫成 `_pct`，確保在 Python DataFrame 欄位索引的安全性。
- Class 名稱請採用大駝峰式 (CamelCase)，並繼承 `BaseFeature`。
- 上方必須掛載 `@register_feature` 裝飾器。

### 腳本結構範本 (Template)：

```python
import pandas as pd
import numpy as np
from ...core import BaseFeature, register_feature
from ...utils import zscore_rolling

@register_feature
class FeatureMorningSmartBuy(BaseFeature):
    name = "f_morning_smart_buy_ewm3_ts_rank"
    description = "這裡寫下清楚的金融與數學演算法邏輯定義"
    
    # required_columns 是指 `preprocessing` 吐出來後，這個特徵所需吃到的「中間變數欄位」
    required_columns = ["StockId", "Date", "top10_MorningBuy", "TotalVol"]
    data_combination = "cross_broker_tick"

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = data.copy()
        
        # 1. 確保 Time-Series 計算的安全性：按時間排序後 reset_index
        df = df.sort_values(["StockId", "Date"]).reset_index(drop=True)
        
        # 2. 計算特徵本體 (善用 numpy, pandas vectorized)
        raw_val = df["top10_MorningBuy"] / (df["TotalVol"] + 1)
        
        # 3. Time-Series 變換 (EWM, Z-Score, Rank)
        # 關鍵：使用 df.groupby(df["StockId"]) 而非 df.groupby(level="StockId")
        # 這樣可以確保在 reset_index 後仍能正確分組
        ewm_val = raw_val.groupby(df["StockId"]).transform(
            lambda x: x.ewm(span=3).mean()
        )
        
        # zscore_rolling 支援 eps 參數，預設 eps=1e-8
        # 對於數值範圍極小的特徵（如 AFD 特徵，raw 值 ~1e-9），應使用 eps=1e-10
        out_series = ewm_val.groupby(df["StockId"]).transform(
            lambda x: zscore_rolling(x, 42)  # 使用預設 eps=1e-8
        )
        
        # 4. 後處理與防呆處理 (清洗無限與空值)
        out = out_series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
        
        # 5. 回傳必須帶有 StockId 與 Date，以及自己的特徵名稱
        return pd.DataFrame({
            "StockId": df["StockId"],
            "Date": df["Date"],
            self.name: out
        })
```

### 🚨 Time-Series 絕對禁忌：
在處理歷史資料平滑 (Rolling / EWM / Shift / Diff) 時，**必須使用 `df.groupby(df["StockId"])` 進行分組**！

```python
# ✅ 正確方式 (columns-based grouping，適用於 reset_index 後的 DataFrame)
raw.groupby(df["StockId"]).transform(lambda x: x.rolling(20).mean())

# ❌ 舊規範方式 (deprecated - 會與 reset_index 衝突)
raw.groupby(level="StockId").transform(lambda x: x.rolling(20).mean())
```

否則可能造成 A 股票的最後一筆資料去計算 B 股票的第一筆資料，造成**致命的 Lookahead 或是資料污染 Bug**。

---

<!-- ============================================================ -->
<!-- 舊規範 (DEPRECATED - 請勿採用)                                -->
<!-- ============================================================ -->

<!--
## [舊] 1. 資料集組合方式 (DEPRECATED)

❌ 舊規範缺少 `cross_broker_daily` 組合

舊組合列表：
- `single_stock_daily`: 僅需日K歷史資料 (`df_daily`)。
- `single_stock_tick`: 需逐筆交易歷史資料 (`df_tick`)。
- `single_stock_broker`: 需分點進出歷史資料 (`df_broker`)。
- `cross_broker_tick`: 同時需要多種異質歷史資料 (如 `df_broker` + `df_tick`) 進行交叉比對。

-->

<!--
## [舊] 3. 特徵腳本編寫規範 (DEPRECATED)

❌ 舊 Template 使用 `set_index` 模式，會與 `reset_index` 衝突

```python
# ❌ 舊規範 (deprecated)
def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df = data.copy()
    
    # 舊規範使用 set_index
    all_features = df.set_index(["StockId", "Date"]).sort_index()

    raw_val = all_features["top10_MorningBuy"] / (all_features["TotalVol"] + 1)
    
    # 舊規範使用 groupby(level="StockId") - 會與 reset_index 衝突
    ewm_val = raw_val.groupby(level="StockId").transform(lambda x: x.ewm(span=3).mean())
    
    out_series = ewm_val.groupby(level="StockId").transform(_ts_rank)
    out = pd.Series(out_series.values, index=all_features.index)
    
    final_result = pd.DataFrame({
        self.name: out
    }).reset_index()
    
    return final_result
```

問題：
1. 使用 `set_index` 後若在其他地方做了 `reset_index`，會導致 `level="StockId"` 無法找到正確的 index level
2. 最後 `reset_index()` 會把所有 columns 包含進 DataFrame，包括不需要的中間變數
3. 不夠直覺，容易出錯
-->

---

## 4. User Input / Output 介面規範 (`api.py`)

`api.py` 扮演著防火牆與控制中心的角色，保證無論使用者餵什麼進來，介面都能消化並吐出標準結果。

- **強制 API 簽名**：`compute_features(data_combination: str, data: pd.DataFrame, **kwargs) -> pd.DataFrame`
- **User Input DataFrame 規範**：
  - User 傳入的主 DataFrame 必須稱為 `data`。
  - 主資料 `data` 中不可預期一定帶有特殊的欄位 (除了必備的 `StockId`, `Date` 外)，因此必須在此層接住防呆。
  - 當需要第二份、第三份多樣貌特徵集，必須將其他資料作為關鍵字參數傳遞：
    - `cross_broker_tick` → `tick_data=df_tick`
    - `cross_broker_daily` → `daily_data=df_daily`
    - 這個關鍵字必須在 `api.py` 以及對應的 preprocessing 中處理容錯。
- **User Output 規範**：
  - 產出的 DataFrame `Date` 欄位建議全數轉化為標準的 `int` (`YYYYMMDD`)，確保 merge 時不出錯。
  - 最終回傳的 DataFrame 必須經過 `utils.ensure_unique_key` 以保證 `(StockId, Date)` 主鍵的絕對唯一性 (去重)。
  - 會同時包含 `StockId`, `Date` 還有**所有該組合計算出來的特徵欄位**。

---

## 5. Feature Test Code (特徵測試與合併)

每當用 LLM 開發出一個全新的 Feature 後，應當通過以下三層手續才算完成合併：

1. **隔離編寫 (Isolation)**：
   先在 `/tmp` 或外部測試資料夾中撰寫一支 `_draft.py`，撰寫你的演算法。
2. **手動資料驗證 (Manual Validation)**：
   自己隨機建立帶有 `[NaN, Inf, 極大值, 極小值, 0]` 的 Numpy Array 模擬資料，確保演算法不會因為除以零噴錯而導致 Pipeline 停止。
3. **放置與路由測實驗證 (Integration & Routing Test)**：
將檔案命名為 `calculate_{feature_name}.py` 放入 `feature_engine/features/{combination_name}/`。
    編寫測試檔 (例如 `test_features_sample.py`)：
    ```python
    # 確保你新加的特徵欄位會精確出現在產出的 DataFrame 中
    # 並且測試時間沒有明顯的退化。
    from feature_engine.api import compute_features
    output = compute_features("你特徵所屬的combination", df_main, **kwargs)
    assert "你的特徵名稱" in output.columns
    ```
4. **提交 (Push)**：
    確認沒有打破原本 `assert_frame_equal` 100% 同步測試後，才代表該特徵正確且順利被合併進入模組中。

---

## 6. 數值穩定性與 zscore_rolling 使用規範

### zscore_rolling 函數參數

```python
from feature_engine.utils import zscore_rolling

# 函數簽名
def zscore_rolling(series: pd.Series, window: int = 20, min_periods: int = None, eps: float = 1e-8) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / (rolling_std + eps)."""
    ...
```

### eps 參數說明

`eps` 參數用於避免當 `rolling_std` 接近 0 時發生除以零的問題。參數選擇原則：

| eps 值 | 適用場景 | 說明 |
|--------|---------|------|
| `1e-8` (預設) | 一般特徵 | 適用於 raw 值範圍在正常數量級（~1e-2 到 ~1e2）的特徵 |
| `1e-10` | 極小值特徵 | 適用於 raw 值極小（~1e-9 到 ~1e-12）的特徵，如 AFD 大氣流體力學特徵 |

### 為什麼 eps 選擇很重要？

當特徵的 raw 值極小時，標準差也會極小：

```python
# 假設 std = 1e-9
# 使用 eps = 1e-8:
zscore = (raw - mean) / (1e-9 + 1e-8) = (raw - mean) / 1.1e-8  # eps 主導！

# 使用 eps = 1e-10:
zscore = (raw - mean) / (1e-9 + 1e-10) = (raw - mean) / 1.1e-9  # std 主導
```

### AFD 特徵範例

```python
# AFD 特徵必須使用 eps=1e-10
out_series = zscore_rolling(df_res['raw'], window=self.ZSCORE_WINDOW, eps=1e-10)
```

### 一般特徵範例

```python
# 一般特徵使用預設 eps=1e-8
out_series = zscore_rolling(raw_val, window=42)  # 預設 eps=1e-8
```
