# `feature_engine` 使用說明文件

這份文件介紹了 `feature_engine` 模組的標準化 API 使用方法。此模組採用了**統一介面 (Unified Interface)** 的設計模式，您不需要手動實例化各種不同的特徵類別，而是透過 `compute_features` 這個統一的函數入口，配合指定不同的 `data_combination` (資料組合類型) 來幫您一鍵算出所有特徵。

## 核心 API 介面

`feature_engine` 提供了三個對外開放的核心函數，都統一放在 `feature_engine.api` 中：

1. `compute_features()`：計算特徵並回傳結果 DataFrame。
2. `list_features()`：列出目前系統中註冊支援的所有特徵名稱。
3. `describe_features()`：列出所有特徵的詳細資訊 (名稱、所屬群組、需要欄位等)。

---

## `compute_features` 函數詳解

這是最核心的計算函數，其定義如下：
```python
def compute_features(
    data_combination: str,
    data: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
```

### 1. `data_combination` (Input 類型)
這個參數告訴系統您傳入的是什麼層級的資料，並且系統要用哪一群特徵模組來做計算。目前支援六種組合：
*   `'single_stock_daily'`：**日線與籌碼**層級特徵。
*   `'single_stock_tick'`：**逐筆明細**層級高頻特徵。
*   `'single_stock_broker'`：**券商進出**層級高頻特徵。
*   `'cross_broker_tick'`：**券商進出 + 逐筆明細**的綜合高頻輪廓特徵 (Cross-join)。
*   `'cross_broker_daily'`：**券商分點 + 日頻 OHLC**的交叉特徵，用於PHY物理現象、Alpha v15生態模擬特徵與Alpha v17遊戲理論特徵(F31-F34)。
*   `'cross_tick_daily'`：**逐筆明細 + 日頻 OHLC**的交叉特徵，用於 Alpha v16 生態模擬特徵與 Alpha v17 遊戲理論特徵。

### 2. `data` (Input 資料)
您要用來計算特徵的原始 `pd.DataFrame` 資料表。
*   所有的 dataframe 都至少需要包含 `StockId` 與 `Date`。
*   如果是跑 `'single_stock_daily'`，需要有收盤價、最高最低、外資投信超買賣等日K資料。
*   如果是跑 `'single_stock_tick'`，需要有逐筆的 `DealTimeSecond`, `DealPrice`, `DealCount`, `PrFlag`。
*   如果是跑 `'single_stock_broker'`，需要有券商分點的 `BrokerId`, `Price`, `BuyQtm`, `SellQtm`。
*   如果是跑 `'cross_broker_daily'`，`data` 放券商分點資料，`kwargs` 需傳入 `daily_data=df_daily`（包含收盤價、最高價、最低價、成交量等）。
*   如果是跑 `'cross_tick_daily'`，`data` 放逐筆明細資料，`kwargs` 需傳入 `daily_data=df_daily`（包含收盤價與開盤價）。

### 3. `feature_names` (選填)
預設為 `None`，代表**計算該 `data_combination` 註冊的所有特徵**。
如果您只想要計算特定的幾支特徵以節省時間，您可以傳入一個 list，例如：`feature_names=["f_sma_20", "f_vol_cv"]`。

### 4. `**kwargs`
額外參數：
*   當您指定 `data_combination = 'cross_broker_tick'` 時，需要傳入 `tick_data=df_tick`。
*   當您指定 `data_combination = 'cross_broker_daily'` 時，需要傳入 `daily_data=df_daily`。
*   當您指定 `data_combination = 'cross_tick_daily'` 時，需要傳入 `daily_data=df_daily`（逐筆明細作為主 `data`，日頻 OHLC 作為 kwargs）。
*   當您指定 `data_combination = 'cross_tick_broker_daily'` 時，需要傳入 `daily_data=df_daily` 和 `tick_data=df_tick`。這是新增的組合，同時需要券商分點、逐筆明細和日頻 OHLC 資料。

### 5. `Output`
回傳一個合併好特徵的 `pd.DataFrame`，保證具有 `StockId` 與 `Date` 兩個欄位，並且每一個 `(StockId, Date)` 組合都是唯一的。

---

## 程式碼整合範例 (Usage Example)

以下提供如何使用 `feature_engine.api` 的常見範例：

```python
import pandas as pd
from feature_engine.api import compute_features, list_features, describe_features

# ==========================================
# 1. 查看支援的所有特徵與其所需欄位 
# ==========================================
# 打印出一個 DataFrame，讓您看清楚所有特徵的定義
df_info = describe_features()
print(df_info.head())

# 只列出 'single_stock_broker' 註冊的所有特徵名稱
broker_features_list = list_features("single_stock_broker")
print(broker_features_list)

# ==========================================
# 2. 準備原始資料
# ==========================================
df_daily = pd.read_parquet("data/daily_data.parquet")
df_tick = pd.read_parquet("data/tick_data.parquet")
df_broker = pd.read_parquet("data/broker_data.parquet")

# ==========================================
# 3. 執行特徵計算 (One-Liner API)
# ==========================================
print("Calculating Daily OHLCV Features...")
# 第一個參數指定情境，第二個傳入對應的 DataFrame
feature_daily = compute_features('single_stock_daily', df_daily)

print("Calculating Tick High-Frequency Features...")
feature_tick = compute_features('single_stock_tick', df_tick)

print("Calculating Broker High-Frequency Features...")
feature_broker = compute_features('single_stock_broker', df_broker)

print("Calculating Cross Profile Features...")
# Cross 的情境下，data 帶 broker，而 kwargs 補上 tick_data
feature_profile = compute_features('cross_broker_tick', df_broker, tick_data=df_tick)

print("Calculating Cross Tick-Daily Features...")
# data 帶逐筆明細，kwargs 補上日頻 OHLC
feature_tick_daily = compute_features('cross_tick_daily', df_tick, daily_data=df_daily)

# ==========================================
# 4. 合併所有特徵 (Merge Features)
# ==========================================
# 每張表出來必定都是 (StockId, Date) 為 unique key，直接 outer merge 即可
final_features = feature_daily.merge(feature_tick, on=["StockId", "Date"], how="outer")
final_features = final_features.merge(feature_broker, on=["StockId", "Date"], how="outer")
final_features = final_features.merge(feature_profile, on=["StockId", "Date"], how="outer")
final_features = final_features.merge(feature_tick_daily, on=["StockId", "Date"], how="outer")

print(f"Feature calculation complete. Shape: {final_features.shape}")
```

### 🚨 架構保護機制優勢
1. **內建前處理 (Preprocessing Layer)：** `compute_features` 內建前處理層 (preprocessing logic)，能夠自動幫您清理原始資料，不用擔心直接丟原始髒資料進去的結果。
2. **防呆驗證：** 計算前，系統會自動透過 `.validate()` 確保您傳入的 `data` 是否具備所有的 `required_columns`。若有缺漏欄位會立即發出精準的錯誤提示。
3. **無痛擴充：** 未來有新的特徵被加入時，只要加上 `@register_feature`，您現有的這套 `compute_features` 都不用改寫，就會**自動算好**新加入的特徵出來給您！
