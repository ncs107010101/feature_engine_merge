> 1. **特徵目錄** — 快速瀏覽所有 318 個特徵
> 2. **計算細節** — 每個特徵使用的算法、算子、參數
> 3. **概念邏輯** — 每個特徵捕捉什麼市場現象，避免重複創建概念相似的特徵

---

## 模組總覽

| 模組檔案 | 內容描述 | 特徵數 | 主要概念領域 |
|----------|----------|--------|--------------|
| `daily_ohlcv.py` | 日頻 OHLCV + 籌碼 + 基本面 | 35 | 價量技術面、估值、散戶/法人行為 |
| `hf_tick.py` | 逐筆成交（tick） | 110 | 微結構動量、VPIN、Bar 特徵、極端價格行為、物理統計、量子資訊 |
| `hf_broker.py` | 券商分點買賣 | 31 | 券商集中度、持久性、資訊熵、量子資訊 |
| `hf_profile.py` | 券商 × 逐筆（交叉） | 32 | 行為差異、熱力學保守功、跨券商分佈、資訊理論、凱利-拉頓姆、生態模擬 |
| `single_stock_tick (PHY)` | 物理現象 (Tick) | 16 | 拓撲缺陷、相變、真空壓力、共振 |
| `single_stock_tick + broker + cross (v7, 25)` | 物理+新物理特徵 (Tick/Broker/Cross) | 25 | MHD 螺旋度、LRT 漲落耗散、Crack 裂紋、SINDy 系統識別、QM 量子測量、POL 極化、GEO 幾何流形、NET 馬可夫、反鐵磁性券商交互、量子測量不相容 |
| `cross_broker_daily (v6, 8)` | 物理現象 (Broker) | 8 | 幾何扭矩、超新星坍縮、量子糾纏 |
| `single_stock_daily (v6, 1)` | 物理現象 (Daily) | 1 | 非輻射動量、耗散能量 |
| `Alpha v8 – single_stock_tick (3)` | 泊松通道特徵 (Tick) | 3 | 大單事件後方向、IAT不對稱、泊松強度 |
| `Alpha v9 – cross_broker_tick (3)` | CTW通用估計器 (Cross) | 3 | 3階Markov LLR、定向資訊z分數、Markov記憶深度差 |
| `Alpha v10 – cross_broker_tick (4)` | NRDF抽象空間特徵 (Cross) | 4 | 因果傳輸率、資訊加權價格位移、短長期KL、歷史偏離極性 |
| `Alpha v11 – cross_broker_tick (3)` | 回饋通道與守恆律 (Cross) | 3 | 回饋利用率skew、MI守恆失衡、廣播通道熵差 |
| `Alpha v12 – single_stock_tick (4)` | 生態模擬特徵 (Tick) | 4 | 長尾散播、Masting 掠食者、Lévy-高斯混合、掠食者配偶限制 |
| `Alpha v13 – single_stock_broker (13)` | 生態模擬特徵 (Broker) | 13 | Allee 臨界、核心源區、累積劑量、R0 動能、免疫屏障、近親繁殖、同類相食、潛伏期跳躍、原發/次發背離、Gamma 異質性、化學劑量失衡、指狀刺透、指狀不穩定 |
| `Alpha v14 – cross_broker_tick (5)` | 生態模擬庇護所效應 (Cross) | 5 | 倒貨價差擴大、低調吸籌、爆發集中度、主力拋售強度、空間混合庇護 |
| `Alpha v15 – cross_broker_daily (7)` | 生態模擬邊界與衰竭效應 (Daily) | 7 | 生產者防禦稀釋效應、創始者效應衰退、棲地退移速度差、掠食者飽食邊界、逆向阿利效應枯竭、邊界獻祭比例、底線防禦 |
| `Alpha v16 – cross_tick_daily (5)` | 生態模擬特徵 (Tick) | 5 | 累積劑量枯竭點、密度依賴性擴散阻尼、雙重閾值非線性躍變、動力學極限環耗散枯竭、斑塊存活赤字 |
| `Alpha v17 – single_stock_tick (9)` | 遊戲理論特徵 (Tick) | 9 | 廉價話語分心、裝傻吸籌/派發、漸進逃離投降、嫉妒代價、金錢模仿陷阱、說服懸念滑落、無知羊群偏誤、前向歸納信號 |
| `Alpha v17 – single_stock_daily (6)` | 遊戲理論特徵 (Daily) | 6 | Level-k散戶收割、解讀失敗懷疑、延遲揭露道德危機、名譽成本信號、戰略性無知需求、道德危機槓桿 |
| `Alpha v18 – single_stock_daily (4)` | 行為經濟學特徵 (Daily) | 4 | 不對稱自信溢出、外團體剝削偏誤、自我驅動錯誤報告、挫折驅動換手 |
| `Alpha v17 – single_stock_broker (3)` | 遊戲理論特徵 (Broker) | 3 | 聚焦點 payoff dominant、專家vs大眾背離、膽小鬼僵局 |
| `Alpha v17 – cross_broker_daily (8)` | 遊戲理論特徵 (Cross) | 8 | 消耗戰鎖碼、同溫層泡泡、二階信念褪散、承諾機制鎖碼、漸進信任升級、過度自信溢價、Pool均衡偽裝、現任者豁免 |
| `Alpha v18 – cross_broker_daily (10)` | 賽局理論/行為經濟學特徵 (Cross) | 10 | 風險藉口倒貨、部分說謊隱蔽賣出、內團體隱蔽合謀、污名混合分散、合作崩潰閾值、內團體偏愛交叉、品格污名迴避、地位調節否認、反社會退出不平等、高昂信號排他鎖碼 |
| `Alpha v17 – cross_tick_daily (1)` | 遊戲理論特徵 (Cross) | 1 | 立即揭露恐慌 |
| `AFD v1 – cross_tick_daily (10)` | 大氣流體力學特徵 (Tick+Daily) | 10 | 冷區對流加熱、融券槓桿點火、前沿價格脫離重心、宏觀微觀斜率共振、風暴螺旋度、高螺旋度濾波、錮囚區渦度、大盤摩擦剪切、最佳阻力係數、河彎效應 |

**通用輸出格式**：所有模組輸出 `(StockId, Date)` 為唯一鍵的 DataFrame，每列代表一支股票一天的特徵值。

---

## 通用算子說明

以下時間序列算子在多個特徵中反覆使用：

| 算子 | 符號 | 說明 |
|------|------|------|
| Rolling Z-Score | `zscore(x, w)` | `(x - rolling_mean(x, w)) / (rolling_std(x, w) + ε)` |
| EWM Smooth | `ewm(x, s)` | 指數加權移動平均，span=s |
| TS Rank | `ts_rank(x, w)` | 在過去 w 天的百分位排名，減 0.5 使中心為 0 |
| TS Z-Score | `ts_zscore(x, w)` | 等同 zscore(x, w)，命名區分用 |
| Cross-sectional Rank | `cs_rank(x)` | 同一天所有股票間的百分位排名 |
| Clip | `clip(x, lo, hi)` | 截斷至 [lo, hi] 範圍 |

---

## 一、Daily OHLCV 特徵（35 個）

**輸入**：日頻合併資料（OHLCV + 籌碼 + 基本面欄位）
**模組**：`features/daily_ohlcv.py`

### A. 基礎技術面（2 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_sma_20` | `rolling(收盤價, 20).mean()` | **20日均線**。衡量中期趨勢方向。 |
| 2 | `f_ema_12` | `ewm(收盤價, span=12)` | **12日指數均線**。比 SMA 更靈敏的趨勢追蹤。 |

### B. 優化日頻特徵（11 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 3 | `f_chip_large_holder_pct` | `ffill(超過1000張集保占比)` | **大戶持股比例**。反映籌碼集中度。 |
| 4 | `f_day_trade_pct` | `當沖買賣占比 / 100` | **當沖佔比**。反映短線投機氛圍。 |
| 5 | `f_turnover_rate` | `成交量 / 流通在外股數` | **換手率**。衡量流動性和交易活躍度。 |
| 6 | `f_vol_cv` | `std(vol, 20) / mean(vol, 20)` | **成交量變異係數**。量能穩定性指標。 |
| 7 | `f_short_change_roc` | `融券餘額3日變動率` | **融券變動**。反映空方力量集結或回補。 |
| 8 | `f_open_gap` | `(開盤 - 前收) / 前收` | **開盤跳空幅度**。反映隔夜資訊衝擊。 |
| 9 | `f_ret_5d` | `pct_change(收盤, 5)` | **5日報酬率**。短期動量或均值回歸。 |
| 10 | `f_intraday_zscore` | `zscore((收-開)/開, 100)` | **日內漲跌幅Z分數**。衡量當天走勢強度。 |
| 11 | `f_close_loc_smooth` | `mean((收-低)/(高-低), 5)` | **收盤位置平滑**。收盤價在日內區間的位置。 |
| 12 | `f_foreign_flow_zscore` | `zscore(外資買賣/vol_ma20, 20)` | **外資流入Z分數**。外資異常買入信號。 |
| 13 | `f_fund_pe` | `本益比(TEJ)` | **本益比**。基本面估值指標。 |

### C. 複合因子（10 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 14 | `f_hl_range_ewm` | `-cs_rank(ewm(振幅, 5))` | **低波動排名**。捕捉壓縮後爆發前兆。 |
| 15 | `f_retail_heat_composite` | `-加權(rank(tr)+rank(dt)+rank(mc))` | **散戶過熱度**。多頭末期反指標。 |
| 16 | `f_value_composite` | `加權(rank(1/PE)+rank(1/PBR))` | **價值因子**。多維度估值分數。 |
| 17 | `f_div_antiretail` | `法人買賣分歧 × 低散戶熱度` | **法人散戶分歧**。主力靜默建倉信號。 |
| 18 | `f_reversal_composite` | `-(日內+跳空+5日報酬Rank)` | **短期反轉因子**。均值回歸預期。 |
| 19 | `f_residual_momentum` | `(ret10 - ret20) / ATR` | **殘餘動能**。排除大盤趨勢後的動能。 |
| 20 | `f_high_low_range_expansion` | `MAD_5d / MAD_20d` | **波幅擴張**。波動蓄勢後爆發偵測。 |
| 21 | `f_heat_lag` | `corr(heat_t-1, ret_t)` | **熱度預測力**。捕捉市場慣性。 |
| 22 | `f_alpha_accel` | `ret 二階差分 Z` | **報酬加速度**。捕捉趨勢拐點。 |
| 23 | `f_chip_concentration` | `HHI zscore` | **臨界籌碼集中**。大戶建倉完成信號。 |

### D. 量價與形態分析 (13 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 24 | `f_contraction_tension` | `(ATR60/ATR5) × log(1+vol)` | **波動收縮蓄勢**。爆發前兆。 |
| 25 | `f_vol_elasticity` | `Non-linear ATR dynamic` | **波動彈性**。捕捉爆發狀態。 |
| 26 | `f_price_migration` | `Σ|dp| / ATR` | **價格遷移距離**。突破強度。 |
| 27 | `f_phase_diff` | `ret5 - ret20/4` | **相位差**。動能偏離趨勢。 |
| 28 | `f_sector_resonance` | `ret5d × vol_ratio` | **量價共振**。動能品質度量。 |
| 29 | `f_limit_edge` | `min(距漲停, 距跌停) × vol` | **漲跌停壓力**。方向性壓力。 |
| 30 | `f_close_to_high_slope` | `slope(CloseRelToHigh)` | **收盤高點趨勢**。控盤強度。 |
| 31 | `f_price_range_contraction_days` | `count(range < 0.5×avg)` | **窄幅波動天數**。能量積累。 |
| 32 | `f_price_volume_corr_20d` | `corr(ret, vol, 20)` | **價量相關性**。量價是否齊揚。 |
| 33 | `f_short_squeeze_potential` | `ShortInterest / RollMax` | **軋空潛力**。高券額比例與低成交量的共振，預示軋空風險。 |
| 34 | `f_vol_accel` | `vol diff2 Z` | **成交量加速度**。量能放大的二階導數，捕捉爆量啟動信號。 |
| 35 | `f_volume_surprise` | `vol / roll_mean(vol, 20)` | **成交量驚奇度**。捕捉異常放量行為。 |

---

## 二、HF Tick 特徵（110 個）

**輸入**：逐筆成交資料（`DealTimeSecond`, `DealPrice`, `DealCount`, `PrFlag`）
**模組**：`features/hf_tick.py`

### A. 基礎微結構信號（4 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_tick_mom` | `buyVol / totalVol` | **逐筆動量**。主動買佔比 Z。 |
| 2 | `f_vpin` | `|buy - sell| / total` | **VPIN**。知情交易概率。 |
| 3 | `f_trade_skew` | `skew(DealCount)` | **成交偏態**。大單出現頻率。 |
| 4 | `f_close_pressure` | `尾盤淨買比 × 量佔比` | **收盤壓力**。尾盤出貨或拉抬。 |

### B. 日內行為偵測（11 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 5 | `f_tk_surge_1min_count` | 漲幅 > 0.5% 次數 | **急漲頻率**。強力掃貨。 |
| 6 | `f_tk_plunge_1min_count` | 跌幅 < -0.5% 次數 | **急跌頻率**。恐慌拋售。 |
| 7 | `f_tk_morning_surge_ret` | 開盤15分 MAX 漲幅 | **早盤衝高幅度**。主力試探。 |
| 8 | `f_tk_high_zone_active_sell_ratio` | 高價區主動賣佔比 | **高價區拋售**。趁高出貨。 |
| 9 | `f_tk_trade_intensity_max` | max(筆數/sec) | **最大成交密度**。衝擊事件。 |
| 10 | `f_tk_large_vwap_dev` | 大單價 - VWAP | **大單溢價買入**。緊急建倉。 |
| 11 | `f_tk_failed_surge_trap` | 急漲後收盤 < VWAP | **拉高失敗陷阱**。誘多出貨。 |
| 12 | `f_tk_dip_buying_under_plunge` | 急跌中的買入比 | **急跌逢低承接**。法人吸籌。 |
| 13 | `f_tk_morning_trap_severity` | 早盤高 - 尾盤低 | **早盤誘多嚴重度**。追高套牢。 |
| 14 | `f_tk_active_sell_above_vwap` | VWAP 以上主動賣 | **高價主動拋售**。穩定出貨。 |
| 15 | `f_tk_large_sell_ratio` | 大額主動賣 / total | **大額出貨比**。主力離場。 |

### C. 多尺度 Bar 微結構 (26 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 16 | `f_volbar_ofi_skew` | Volume Bar OFI skew | **量能Bar訂單流偏態**。 |
| 17 | `f_volbar_vwap_trend` | Volume Bar VWAP slope | **量能Bar趨勢**。價格推升。 |
| 18 | `f_dollarbar_volatility` | Dollar Bar ret std | **金額Bar波動率**。微結構波動。 |
| 19 | `f_timebar_am_pm_divergence` | AM-PM 買壓差 | **上下午買壓分歧**。出貨型態。 |
| 20 | `f_tk_smart_money_late_vwap_ratio` | 尾盤 VWAP 以上買入 | **尾盤聰明錢**。秘密建倉。 |
| 21 | `f_tk_buy_sell_trade_size_ratio` | avg_buy / avg_sell | **買賣均筆差異**。資訊優勢。 |
| 22 | `f_tkb_variance_ratio` | VR(10) | **方差比**。效率與趨勢偵測。 |
| 23 | `f_tkb_run_length_max` | max sequential buy | **長 Run 買入**。持續掃貨。 |
| 24 | `f_tkb_reversal_frequency` | Direction flips ratio | **價格翻轉頻率**。做市商主導。 |
| 25 | `f_vb_large_trade_domination` | Max Trade / Bar | **大單主導度**。大戶控制。 |
| 26 | `f_vb_active_sell_intensity_var` | Var(ActiveSell) | **主動賣壓波動**。策略出貨。 |
| 27 | `f_vb_pinning_probability` | ret=0 ratio | **價格釘住機率**。吸收籌碼。 |
| 28 | `f_vb_ofi_volatility` | std(OFI) | **訂單流波動**。多空平衡不穩定。 |
| 29 | `f_vb_vwap_twap_deviation` | VWAP/TWAP - 1 | **追漲偵測**。量價配合不對稱。 |
| 30 | `f_vb_close_to_vwap_deviation` | Close / VWAP - 1 | **收盤強勢度**。相對於日內重心的偏離。 |
| 31 | `f_vb_extreme_return_duration_ratio` | Duration(extreme) / avg | **極端走勢持續比**。演算法衝擊。 |
| 32 | `f_vb_trade_size_dispersion` | CV(trade count) | **筆數分散度**。委託拆分偵測。 |
| 33 | `f_vb_absorption_ratio` | Vol(ret=0) / total | **吸收率**。被動建倉/出貨強度。 |
| 34 | `f_db_trend_rsquared` | R² of path | **金額Bar趨勢 R²**。單邊市穩定度。 |
| 35 | `f_db_vpin_proxy` | |B-S|/Total in $Bar | **Dollar Bar VPIN**。 |
| 36 | `f_db_downside_deviation` | RMSD- | **下行偏差**。尾部風險。 |
| 37 | `f_db_peak_velocity_ratio` | max vol / avg | **尖峰速率比**。突發事件。 |
| 38 | `f_db_high_velocity_volatility` | Volatility in fast bars | **高速Bar波動率**。高頻衝擊。 |
| 39 | `f_flow_explosion` | |B-S| × √N | **訂單流爆發**。方向性壓力爆發。 |
| 40 | `f_liquidity_gap` | spread × volatility | **流動性缺口**。大幅跳空預兆。 |
| 41 | `f_order_burst` | burst_freq / √N | **爆量交易頻率**。脈衝式交易。 |

### D. 物理統計特性 (Alpha v4) (26 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 42 | `f_breakout_velocity` | (H-L)/dt | **區間突破速度**。脫離盤整之快。 |
| 43 | `f_order_density` | Vol(@VWAP±2%) | **VWAP 成交密度**。均衡態 vs 逃逸態。 |
| 44 | `f_momentum_continuity` | up_ticks / total | **動能持續性**。價格推動連貫性。 |
| 45 | `f_density_ratio` | vol_peak / tick_peak | **成交密度比**。大單 vs 碎片。 |
| 46 | `f_eth_nonadiabatic_entropy_rate` | dS/dt | **非平衡熵產生率**。遠離平衡態的速度。 |
| 47 | `f_eth_adiabatic_maintenance` | Σ(p-vwap)dx | **絕熱熵維持力**。維持分佈所需的外部功率。 |
| 48 | `f_eth_limit_cycle_dissipation` | ∮P dV | **極端環耗散能**。震盪區間的能量損耗。 |
| 49 | `f_eth_active_translational_dissipation` | v²/μ | **活性位移耗散**。主動位移的熱力學代價。 |
| 50 | `f_flu_instanton_escape_likelihood` | exp(-S) | **瞬子逃逸概率**。局部區間爆炸式突破。 |
| 51 | `f_flu_fluctuation_asymmetry` | skew(flow) | **漲跌波動非對稱**。情緒單邊化演化。 |
| 52 | `f_flu_tur_bound_saturation` | TUR saturation | **熱力學不確定關係飽和度**。市場效率。 |
| 53 | `f_flu_time_asymmetry` | ⟨dx³⟩ | **時間反演對稱破缺**。不可逆趨勢。 |
| 54 | `f_flu_detailed_balance_breaking` | J matrix | **細節平衡破缺**。循環流動性與非平衡。 |
| 55 | `f_att_quasipotential_well` | -log P | **擬勢能井深度**。支持強度與穩定度。 |
| 56 | `f_att_saddle_point_barrier` | peak(Pot) | **鞍點能壘高度**。突破阻力所需的能量。 |
| 57 | `f_att_attractor_transition_rate` | flip count | **吸引子跳變率**。價位中心切換頻率。 |
| 58 | `f_att_min_entropy_deviation` | path error | **最小熵路徑偏離**。無效波動與操縱偵測。 |
| 59 | `f_att_phase_space_contraction` | div(v) | **相空間收縮率**。確定性增加與收斂。 |
| 60 | `f_mac_temperature_mobility` | Var/Mean | **市場有效溫度**。流動性敏感度。 |
| 61 | `f_mac_kramers_moyal_error` | KM error | **KM 預測誤差**。非馬可夫與記憶效應。 |
| 62 | `f_mac_probability_velocity_circulation` | ∮v·dl | **概率流環量**。價量空間旋渦（洗盤）。 |
| 63 | `f_dpt_phase_transition_singularity` | (T-Tc)^-γ | **相變奇點指標**。集體行動與泡沫臨界。 |
| 64 | `f_dpt_nonreciprocal_interaction` | A-At | **非互易交互作用**。不對稱心理反饋。 |
| 65 | `f_dpt_arrhenius_escape_rate` | Esc rate | **阿瑞尼斯逃逸速率**。跳出箱體的速度。 |
| 66 | `f_dpt_surprisal_derivative` | d(-log P) | **驚訝度變率**。預期瞬間修正。 |
| 67 | `f_dpt_subexponential_responsiveness` | Nonlinear F | **次指數響應度**。極端敏感性。 |

### E. 量子資訊與相對論 (Alpha v5) (20 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 68 | `f_qit_arnold_web_resonance` | Resonance index | **阿諾德網共振**。測量相空間中的共振軌道，捕捉價格在特定中樞的穩定性與逃逸勢能。 |
| 69 | `f_qit_bremsstrahlung_tick_radiation` | Radiation ratio | **制動輻射碎單強度**。價格急跌減速時的碎單佔比，衡量市場在衝擊下的「輻射」損耗。 |
| 70 | `f_qit_decoherence_spin_echo` | Echo consistency | **自旋迴聲去相干**。回測價格回調後的狀態重構保真度，識別假摔與真跌。 |
| 71 | `f_qit_dense_coding_capacity` | Channel capacity | **密集編碼資訊量**。衡量跨檔大額成交的資訊傳輸速度，識別具備預期差的高品質流向。 |
| 72 | `f_qit_dfs_isolation_score` | DFS score | **DFS 空間隔離度**。在去相干無噪空間中的價格穩定性，反映點差內的靜默成交強度。 |
| 73 | `f_qit_flow_embezzlement_proxy` | Proxy value | **訂單流盜用指標**。識別微觀層面上的小單推升效率，偵測潛在的主力盜用流動性行為。 |
| 74 | `f_qit_gie_entanglement_witness` | Witness S | **重力誘導糾纏**。見證價量之間的非古典相關性，量能越大對價位的「重力」糾纏越強。 |
| 75 | `f_qit_gravito_magnetic_induction` | B-field proxy | **重力磁感應動量**。資金流動產生的磁矩效應，衡量價格對資金流向的感應強度。 |
| 76 | `f_qit_hawking_evaporation_rate` | Evap rate | **霍金蒸發率**。在價格高點邊界的賣壓滲透率，捕捉泡沫頂端的能量流失。 |
| 77 | `f_qit_holevo_capacity_utilization` | Capacity utilization | **霍萊沃信道利用率**。衡量訂單流中資訊的利用效率，越高代表市場對新資訊反應越充分。 |
| 78 | `f_qit_incompatible_observables_product` | Non-comm product | **不相容量子觀測積**。根據海森堡不確定性原理，衡量位置與動量的非對易性，識別不穩定均衡。 |
| 79 | `f_qit_lyapunov_ballistic_duration` | Max duration | **彈道傳播持續時間**。主動性買盤維持純淨單向推進的最大時長，衡量趨勢的純度。 |
| 80 | `f_qit_magic_state_injection_ratio` | Magic ratio | **魔術態注入比例**。掃透委託簿的非克隆單比例，強化趨勢的非線性轉折力。 |
| 81 | `f_qit_majorization_deficit` | Deficit | **馬約化排序赤字**。買賣單分佈在不同時段的集中度赤字，反映市場態的分佈不均。 |
| 82 | `f_qit_otoc_scrambling_rate` | OTOC rate | **OTOC 資訊擾動率**。資訊在訂單簿中擴散與混亂的演化速度，衡量系統的混沌程度。 |
| 83 | `f_qit_quantum_zeno_freezing` | Freezing prob | **量子芝諾凍結**。因高頻觀察（頻繁測試價位）而產生的價格釘住效應，反映強大心理障礙。 |
| 84 | `f_qit_subadditivity_entropy_gap` | Entropy gap | **熵次加成性裂口**。價量空間的聯合資訊盈餘，越高代表價量交互產生的額外信號越多。 |
| 85 | `f_qit_teleportation_fidelity` | Fidelity | **狀態傳態保真度**。衡量開盤與尾盤微觀狀態的相似度，識別跨時段的狀態重構。 |
| 86 | `f_qit_virasoro_worldsheet_area` | Worldsheet area | **弦論世界面面積**。路徑包圍的有效能量區域，描述價格運動在複流形上的拓樸複雜度。 |
| 87 | `f_qit_wigner_negativity_depth` | Negativity depth | **維格納負深度**。相空間中 Wigner Function 的負值深度，代表純粹的非古典價量背離。 |

### F. 進階微結構與情緒補充 (23 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 88 | `f_belief_shock` | Shock response | **信念衝擊**。大單後反向移動反映。 |
| 89 | `f_collapse_direction` | Late median dev | **尾盤崩塌方向**。重心位移。 |
| 90 | `f_contextual_momentum_conflict` | Multi-bar conflict | **動能維度衝突**。趨勢不一致。 |
| 91 | `f_early_late_volume_ratio` | AM / PM Vol | **早尾盤量能比**。資訊消化與建倉。 |
| 92 | `f_integer_level_density` | Int price vol | **整數價位吸引力**。心理阻力/支撐。 |
| 93 | `f_intraday_reversal_depth` | V-turn depth | **日內 V 轉深度**。洗盤後拉升。 |
| 94 | `f_intraday_volatility_clustering` | AC(volatility) | **波動集群性**。臨界不穩定。 |
| 95 | `f_large_trade_timing` | HHI(trade time) | **大單時間規律性**。計劃操作。 |
| 96 | `f_order_flow_momentum` | ΣOFI slope | **訂單流動能斜率**。主動方推進。 |
| 97 | `f_price_bimodality` | Bimodal coeff | **價格雙峰震盪**。雙重吸引子。 |
| 98 | `f_price_return_asymmetry` | Log(t+/t-) | **漲跌時長不對稱**。恐慌 vs 軋空。 |
| 99 | `f_spread_compression` | Delta Spread | **流動性壓縮度**。暴風雨前寧靜。 |
| 100 | `f_tail_session_ofi_zscore` | Late 30m OFI Z | **尾盤訂單流異常**。定向衝擊。 |
| 101 | `f_tick_entropy_rate` | sequence entropy | **交易規律熵**。模式可預測性。 |
| 102 | `f_tick_run_imbalance` | MaxRun diff | **連續推動力失衡**。連貫性差異。 |
| 103 | `f_trade_quantization` | Modulo offset | **交易量量子化偏移**。模式突變。 |
| 104 | `f_volume_acceleration` | d²V/dt² (Late) | **量能隧道效應**。突破二階強度。 |
| 105 | `f_volume_price_divergence` | corr(V, P) segment | **價量分歧度**。背離預警。 |
| 106 | `f_vwap_gravity_pull` | Gravity coeff | **VWAP 重力拉力**。均值回歸慣性。 |
| 107 | `f_critical_density_ratio` | Density ratio | **臨界交易密度比**。臨界吞吐力。 |
| 108 | `f_liquidity_gap_intensity` | Gap intensity | **流動性缺口強度**。跳動趨勢。 |
| 109 | `f_price_breakout_velocity` | Breakout v | **價格突破速度**。衝擊速率。 |
| 110 | `f_second_level_momentum` | Sec momentum | **秒級動量連續性**。高頻趨勢。 |

---

## 三、HF Broker 特徵（31 個）

**輸入**：券商分點買賣資料（`BrokerId`, `BuyQtm`, `SellQtm`）
**模組**：`features/hf_broker.py`

### A. 券商分佈與一致性 (26 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_broker_entropy` | -Σ p log p | **券商分佈資訊熵**。集中度 (取反)。 |
| 2 | `f_top5_consist` | Top5 overlap | **主力券商一致性**。機構持倉行為。 |
| 3 | `f_broker_persist` | Continuous dir | **券商方向持久性**。市場共識強度。 |
| 4 | `f_activ_surp` | Count / Mean | **活躍度突變**。散戶湧入偵測。 |
| 5 | `f_flow_diverg` | buy_N / total_N | **資金流分歧**。多空對決共識。 |
| 6 | `f_net_reversal` | (Net - Mean) / Std | **淨買超反轉**。主力態度轉變。 |
| 7 | `f_broker_hhi_buy_sell_diff` | HHI_b - HHI_s | **買賣集中度差異**。建倉信號。 |
| 8 | `f_broker_retail_participation` | N_active | **市場參與券商數**。散戶熱情指標。 |
| 9 | `f_broker_daytrade_intensity` | Σmin(b,s)/Total | **券商當沖強度**。日內來回交易。 |
| 10 | `f_broker_conviction_buyer_ratio` | C_buy | **買方集中度**。堅定看多信念。 |
| 11 | `f_broker_conviction_seller_ratio` | C_sell | **賣方集中度**。法人集體出貨。 |
| 12 | `f_broker_newcomer_ratio` | Newcomers / Total | **新進者力道**。新大戶建倉。 |
| 13 | `f_broker_loyal_inflow` | Loyal ratio | **忠誠資金流入**。堅定買家強度。 |
| 14 | `f_broker_polarization_index` | Std(NetRatio) | **極化指數**。多空分歧激烈。 |
| 15 | `f_broker_hhi_concentration` | Z(N_active) | **券商集中度變動**。結構突變。 |
| 16 | `f_newcomer_net_momentum` | Low_broker VWAP dev | **小戶成本偏離**。反向指標。 |
| 17 | `f_smart_money_accel` | d²Net / dt² | **主力加速度**。力道變化。 |
| 18 | `f_energy_integral` | ΣNet dτ | **淨買能量積分**。近期蓄能程度。 |
| 19 | `f_direction_consistency` | Continuous dir score | **方向一致性**。操作決心。 |
| 20 | `f_distribution_energy` | Freq × log(Int) | **倒貨能量**。高頻出貨預警。 |
| 21 | `f_mac_field_current_divergence` | div J | **場電流發散度**。籌碼鬆動 vs 收籌。 |
| 22 | `f_mac_broker_connectivity_breakdown` | Loss coeff | **網路連通崩潰度**。共識瓦解。 |
| 23 | `f_broker_concentration_shift` | ΔHHI | **集中度遷移**。向少數券商集中（相變）。 |
| 24 | `f_broker_herding_intensity` | |2p-1| | **券商羊群效應**。集體狂熱或恐懼。 |
| 25 | `f_consensus_fracture` | Trend/Rev ratio | **共識斷裂度**。主力對決臨界。 |
| 26 | `f_net_buy_persistence_slope` | slope(Net) | **淨買慣性斜率**。建倉慣性。 |

### B. 量子資訊與物理 (Alpha v5) (5 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 27 | `f_qit_athermal_free_energy` | KL div | **非熱力學自由能**。衡量券商分佈偏離均勻態的自由能密度，捕捉主力突發性的集中行為。 |
| 28 | `f_qit_dilution_factor_inverse` | 1/Dilution | **參與態稀釋因子倒數**。衡量參與者結構的集中度與稀釋程度，捕捉市場參與者的結構性變遷。 |
| 29 | `f_qit_information_causality_bound` | Information logic | **資訊因果界限**。基於資訊因果理論，衡量券商間資訊溢出的上限，識別具備前瞻性的資金動態。 |
| 30 | `f_qit_liquidity_catalysis_ratio` | Catalysis % | **流動性催化率**。造市類券商對成交量的催化作用，衡量不推升價格的「催化」成交比率。 |
| 31 | `f_qit_topological_volume_defects` | Defect density | **拓樸量能缺陷**。識別成交價量雲圖中的拓樸缺陷點，捕捉均價偏差引發的結構性吸籌/倒貨。 |

---

## 四、HF Profile 特徵（32 個）

**輸入**：券商買賣 + 逐筆成交（交叉合併）
**模組**：`features/hf_profile.py`

### A. 市場剖面與行為 (17 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_smart_accumulation_blwVWAP` | Top10% @ <VWAP | **聰明錢低價吸籌**。機構隱蔽建倉。 |
| 2 | `f_retail_trapped_top20` | Bot80% @ >Top20% | **散戶追高被套**。見頂信號。 |
| 3 | `f_retail_falling_knife_bot20` | Bot80% @ <Bot20% | **散戶接刀**。恐慌撿便宜。 |
| 4 | `f_lateday_conviction` | Top20% @ LateDay | **大戶尾盤確信**。高確信度建倉。 |
| 5 | `f_smart_breakout_top20` | Top10% @ >Top20% | **大戶突破買入**。機構看好突破。 |
| 6 | `f_morning_retail_sel_bot_80%` | Bot80% @ Morning | **散戶早盤拋售**。反向指標。 |
| 7 | `f_morning_smart_buy_top_10%` | Top10% @ Morning | **大戶早盤建倉**。搶先佈局。 |
| 8 | `f_large_psell_wall_top_5%` | Top5% PassiveSell | **大戶掛單賣牆**。上方阻力。 |
| 9 | `f_aggressor_concentration` | HHI_b - HHI_s | **主動方集中度差**。攻守勢力。 |
| 10 | `f_morning_smart_active_buy` | Top10% AM ActiveBuy | **大戶早盤扫貨**。搶手行為。 |
| 11 | `f_morning_retail_panic` | Bot80% AM PanicSell | **散戶早盤恐慌**。原始強度。 |
| 12 | `f_large_passive_sell_wall` | Top10% PassiveSell | **大戶被動賣牆**。阻力。 |
| 13 | `f_large_broker_active_buy` | Top10% ActiveBuy | **大券商主動進攻**。最強多頭。 |
| 14 | `f_large_broker_passive_buying` | Top10% PassiveBuy | **大券商被動承接**。下方支撐。 |
| 15 | `f_concentrated_passive_buy` | HHI(PassiveBuy) | **被動買入集中度**。隱蔽吸籌。 |
| 16 | `f_eth_conservative_work` | NetBuy / Path | **保守功發散度**。吸籌效率。 |
| 17 | `f_broker_sync_intensity` | SyncRate | **隱蔽協同買入**。背後關聯。 |

### B. 截面與跨券商分析 (3 個)
 
| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 18 | `f_flow_deviation` | Flow vs Return Dev | **量價流向偏離**。資金流向與報酬率的非線性背離程度。 |
| 19 | `f_retail_trap` | Retail Trap Index | **散戶陷阱深度**。散戶高位承接後被套牢的空間與時間加權值。 |
| 20 | `f_cost_stack` | Smart Money Cost | **主力成本堆疊**。主力資金在不同位階的持倉權重，形成支撐/壓力牆。 |

### C. 資訊理論特徵 (8 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 21 | `f_it_secrecy_capacity_rate` | `(I(X;Y_large) - I(X;Y_small)) / H(T) * net_ratio_big` | **安全容量率**。大戶與散戶對知情交易的貢獻差異，標準化後衡量訊息傳遞效率。 |
| 22 | `f_it_holevo_privacy_premium` | `χ_up - χ_dn` | ** Holevo 資訊溢價**。上漲/下跌環境下的大戶資訊內容差異，捕捉主力對方向的隱藏信念。 |
| 23 | `f_it_eavesdropper_confusion_polarity` | `D_KL(P_small || P_big) * net_ratio_big` | **竊聽者困惑度**。散戶與大戶行為分佈的 KL 散度，衡量散戶被誤導的程度。 |
| 24 | `f_it_public_private_kl_direction` | `KL(Public || Private) * direction` | **公私訊息背離**。掛單偏度與成交偏度的 KL 散度，乘以方向性因子衡量訊息不對稱。 |
| 25 | `f_it_directed_info_updown_asymmetry` | `DI_up - DI_down`, rolling z-score | **定向資訊上下行不對稱**。衡量券商流向對漲跌方向的定向資訊差異，反映訊息傳遞的方向性偏差。 |
| 26 | `f_it_causal_phi_coefficient` | `φ(B, Y)` in 20-day window | **因果 φ 係數**。券商方向與成交方向的 phi 相關係數，衡量因果關聯強度。 |
| 27 | `f_it_np_logloss_differential` | `Gain_up - Gain_dn` | **非參對數損失差**。券商買入時對漲的增益與賣出時對跌的增益之差，捕捉非對稱預測能力。 |
| 28 | `f_it_time_reversed_di_gap` | `(DI_fwd - DI_rev) / (DI_fwd + DI_rev) * ewm_net` | **時間反演定向資訊差**。衡量資訊流的不可逆程度，乘以動量權重反映趨勢持續性。 |

### D. Group 4 凱利與拉頓姆特徵 (4 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 29 | `f_it_causal_kelly_growth` | `f* - f_base`, 20日滾動窗口 | **因果凱利增量**。衡量條件凱利（ given 今日券商方向）與基準凱利之差，反映券商對漲跌的因果影響力。 |
| 30 | `f_it_lautum_penalty_polarity` | `-D_KL(P(B|Y_lag)||P(B)) * net_ratio`, 20日窗口 | **拉頓姆懲罰極性**。KL 散度衡量券商分布偏離均勻態的程度，乘以淨買比反映主力隱蔽行為。 |
| 31 | `f_it_compression_mismatch_redundancy` | `NMI × tanh(skew_B × skew_T)` | **正規化壓縮冗餘方向**。NMI 衡量券商與成交方向的資訊壓縮效率，乘以偏態積反映量價不對稱程度。 |
| 32 | `f_it_conditional_entropy_compression_gain` | `Gain_up - Gain_dn` | **條件熵壓縮增益差**。上漲/下跌環境下的對數增益差，捕捉券商方向對價格變動的預測能力。 |

---

## 五、Alpha v6 Physics (PHY) 特徵 (25 個)

**輸入**：逐筆成交 (`single_stock_tick`)、日頻 (`single_stock_daily`)、券商日頻 (`cross_broker_daily`)
**模組**：`features/single_stock_tick/`, `features/single_stock_daily/`, `features/cross_broker_daily/`

### A. 逐筆級物理特徵 (16 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_phy_casimir_vacuum_pressure_gradient` | `Σ(TickRet / VolRatio)` | **卡西米爾真空壓力**。衡量無交易量時價格的漂移邊際。 |
| 2 | `f_phy_vacancy_induced_gapless_mode` | `Σ(VolRatio * sign(Ret) * Ret²)` | **空位誘導能隙**。衡量大單在價格空洞區產生的非對稱衝擊。 |
| 3 | `f_phy_wignersmith_phase_derivative_q` | `Ret * VolRatio * Mask` | **維格納-史密斯相位導數**。捕捉趨勢轉折處的相位時滯。 |
| 4 | `f_phy_dcdw_collinear_acceleration` | `0.5 * (Flow*Ret + |Flow*Ret|) * sign(Ret)` | **共線加速度**。主動性買盤對趨勢的單邊推動力。 |
| 5 | `f_phy_dislocation_glide_climb_bias` | `log(R_dn / R_up)` | **位錯滑移偏置**。买卖壓力不對稱引發的趨勢斜率偏置。 |
| 6 | `f_phy_neutrino_energy_degradation_regeneration` | `Inertia * IntraRet` | **中微子能量再生**。價格在日內大幅波動後的動能恢復強度。 |
| 7 | `f_phy_tidal_trapping_breakout` | `Trap * IntraRet` | **潮汐陷阱突破**。價格長時間糾結於 VWAP 後的爆发力。 |
| 8 | `f_phy_fractional_charge_excitation` | `VolRatio * ConsecTicks * sign` | **分數量子激發**。同一價位堆疊密集後的能量釋放。 |
| 9 | `f_phy_phase_matching_resonance` | `|AutoCorr| * NetBuyRatio` | **相位匹配共振**。分鐘級訊號與日級資金流的頻率共振。 |
| 10 | `f_phy_ashtekar_torsion_axial_current` | `Torsion * VolRatio` | **阿什特卡扭率軸流**。價格相對於買賣價差的非線性扭轉強度。 |
| 11 | `f_phy_chiral_parity_violating_flux` | `dSpread * VolRatio * sign` | **手性宇稱不守恆通量**。價差變動與主動流向的非對稱耦合。 |
| 12 | `f_phy_parity_transformation_invariant` | `dSpread * sign * VolRatio * 1000` | **宇稱變換不變量**。捕捉市場在對稱操作下的價格阻尼。 |
| 13 | `f_phy_meromorphic_zero_dominance` | `Σ(MidRet * dt)` | **亞純零點主導**。中間價微幅跳動的累積時間效應。 |
| 14 | `f_phy_scattering_vs_medium_reservoir_flux` | `VolRatio * sign * QuoteRet` | **散射與儲層通量**。報價變動與成交流向的微觀散射強度。 |
| 15 | `f_phy_optical_depth_breakthrough` | `Σ(DepthRatio * TickRet)` | **光學深度突破**。突破委託簿「密度」後的加速效應。 |
| 16 | `f_phy_landau_peierls_quench` | `RevFreq * VolRatio * sign` | **朗道-派爾斯驟冷**。急漲跌反轉後的能量耗散。 |

### B. 券商級物理特徵 (8 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 17 | `f_phy_bright_dark_mode_covariance` | `-corr(Top10, Bot80) * Top10` | **亮暗模態協方差**。大戶與散戶的負相關同步程度。 |
| 18 | `f_phy_dark_em_repulsive_force` | `-(Bot80 - Top10) * |Ret|` | **暗電磁斥力**。散戶與大戶極化後的反向排斥力。 |
| 19 | `f_phy_entanglement_generation_trajectory` | `corr(Top5, Ret) * Top5` | **糾纏生成軌跡**。主力資金與價格報酬的非區域性糾纏。 |
| 20 | `f_phy_geodesic_jacobi_convergence` | `sign * |CostAdv * NBR|` | **測地線雅可比收斂**。主力成本優勢引發的趨勢收斂強度。 |
| 21 | `f_phy_geometric_torque_scattering` | `Top5_NBR * G` | **幾何扭矩散射**。價格逼近邊界時的高斯曲率吸引力。 |
| 22 | `f_phy_interstitial_topological_stress` | `BuyRat*(1-Pos) - SellRat*Pos` | **間隙拓撲應力**。大戶在極端價位的被動支撐與壓制壓力。 |
| 23 | `f_phy_multiplex_coherence_alignment` | `Top10_NBR * |Score|` | **多層相干對齊**。多週期趨勢共振與主力同步度。 |
| 24 | `f_phy_supernova_core_collapse_vector` | `-(Bot80 - Top10) * PriceDev` | **超新星核坍縮矢量**。散戶堆疊導致的重心失穩坍縮。 |

### C. 日頻物理特徵 (1 個)

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 25 | `f_phy_bic_radiationless_momentum` | `(VolRat / HL_Ret) * IntraRet` | **非輻射動量**。排除震盪損耗後的有效推進動量。 |

---

## 六、Alpha v7 新物理特徵（25 個）

**輸入**：逐筆成交（`single_stock_tick`）、券商分點（`single_stock_broker`）、交叉（`cross_broker_tick`）
**模組**：`features/single_stock_tick/`, `features/single_stock_broker/`, `features/cross_broker_tick/`
**Batch**：20260312 整合批次

### A. MHD 磁流體力學（3 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_mhd_magnetic_helicity` | `A=cumsum(NetFlow)`, `B=tanh(ΔP×10)`, `H=ΣA·B / (N·max|A|)` | **訂單流磁螺旋度**：訂單流動量累積（A）與價格方向（B）的螺旋纏繞度，正值=持續上漲方向性強。 |
| 2 | `f_mhd_beltrami_alignment` | `cosine_sim(B_bin, J_bin_diff)` per 50-tick bin | **貝特拉米無力場對齊**：50-tick bin 的價格趨勢（B）與大單加速度（J）的 cosine 相似度，越接近 1 越無耗散。 |
| 3 | `f_mhd_vortex_asymmetric_contraction` | `log(det_cov_up / det_cov_dn)` in (ΔP_norm, Vol_norm) | **渦旋相空間不對稱收縮**：上漲/下跌 tick 在正規化相空間中協方差行列式的對數比，捕捉動態不對稱性。 |

### B. PVG 配分函數與自旋玻璃（3 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 4 | `f_pvg_negative_temperature` | `corr(|P-Open|, NetFlow) × confinement_60d` | **負溫度微結構反轉**：能量-狀態相關性 × 60 日成交量受限度，相關反轉 = 負溫度信號。 |
| 5 | `f_pvg_spin_glass_susceptibility` | `Σ autocorr(spin_bin, lag) × mean_spin` per 100-tick | **自旋玻璃磁化率**：100-tick bin 的淨流方向（自旋）自相關之和 × 平均方向，衡量凍結程度。 |
| 6 | `f_pvg_high_energy_reference_shift` | `log(vol_above_E0 / vol_below_E0)`, E0=large-lot VWAP | **極端能量參考偏移**：以大單（≥75th）VWAP 為能量基準 E0，計算 E0 上下成交量的對數比值。 |

### C. LRT 線性響應理論（3 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 7 | `f_lrt_fd_asymmetry` | `(R_up-R_dn)/(R_up+R_dn)`, `R=|mean|/std` of `dp/vol` | **漲落耗散非對稱**：上漲/下跌 tick 的 dp_per_vol 響應效率比，正=上漲效率高。 |
| 8 | `f_lrt_kubo_response_polarization` | 大單（≥90th）後 5-tick 響應 `Δp/vol`，比較 buy vs sell | **庫柏響應極化度**：大買/賣單衝擊後的 5-tick 價格響應比較，量化市場對主動性的對稱性。 |
| 9 | `f_lrt_conjugate_coskewness` | `E[(V-μV)²(P-μP)]/(σV²·σP)` per 50-tick bin | **共軛變數不確定性偏態**：bin 成交量與 VWAP 偏離的 Co-Skewness，揭示量價耦合的非線性。 |

### D. CRACK 裂紋機制（3 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 10 | `f_crack_creep_to_jump` | 上/下吃檔量 30-tick bin 的二階差分（加速度）差 | **潛變至斷裂臨界**：市場在累積壓力後突然跳變，正值=向上加速。 |
| 11 | `f_crack_time_reversal_action` | `(S_fwd-S_rev)/(S_fwd+S_rev)×sign(Close-Open)` | **時間反演路徑作用量差**：正向時間序列與反演序列的差，×方向，不對稱=不可逆趨勢。 |
| 12 | `f_crack_liquidity_collapse` | BuyPr/SellPr median per 50-tick bin → advance/retreat ratio | **流動性塌縮指標**：賣方撤退比買方撤退更多=正值，代表流動性向上塌縮。 |

### E. SINDy 稀疏識別（3 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 13 | `f_sindy_drift_polarity` | `mean_ret/std_ret` per 30-tick bin（ret=ΔP/P_lag） | **SINDy 漂移項極性**：日內收益率均值 t-統計量，捕捉系統性的內生趨勢方向。 |
| 14 | `f_sindy_residual_asymmetry` | `skew(dp - EWMA_pred(dp, span=20))` per 20-tick bin | **系統識別殘差非對稱性**：EWMA 預測殘差偏態，偏離高斯分佈揭示隱蔽驅動力。 |
| 15 | `f_sindy_phase_transition_gain` | `KL(Gauss_1st_half ‖ Gauss_2nd_half) × sign(μ2−μ1)` | **相變奇點資訊增益**：日內前後分佈的 KL 散度乘以方向，量化市場制度轉換幅度。 |

### F. QM 量子測量（3 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 16 | `f_qm_coarse_graining_loss_bias` | `(down_info_loss - up_info_loss) / total_disp` per 5-min bar | **粗粒化資訊損失偏態**：向下運動比向上運動「損失」更多 tick-level 精細資訊=跌勢更隱蔽。 |
| 17 | `f_qm_density_offdiag_relaxation` | 大/小流量 bin 的交叉相關衰減率 × sign(C_lag1)，50-tick bin | **密度矩陣非對角項退相干**：大單與小單流量相關性的衰減速率，衡量退相干快慢。 |
| 18 | `f_qm_measurement_incompatibility` | `A_z × (1 − |B_z|)`，A_z=大戶淨流 z-score，B_z=tick 買比 z-score | **量子測量不相容**：大戶主導性（A）與市場追價意願（B）無法同時測量的互補性。 |

### G. POL 極化子動力學（3 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 19 | `f_pol_effective_mass_asymmetry` | `(M_dn−M_up)/(M_dn+M_up)`, `M=Σvol/Σ|ΔP|` | **有效質量非對稱**：下跌慣性（質量）大於上漲=正值，捕捉方向性阻力差異。 |
| 20 | `f_pol_spectral_centroid_shift` | FFT(momentum, 20-tick bin) 的頻率重心 × sign(Close−Open)，rolling-20d z-score | **動量場頻譜重心偏移**：日內動量場的主導頻率 × 方向，高頻 = 短週期震盪主導。 |
| 21 | `f_pol_phonon_drag_coefficient` | 大單（≥90th）後 10 筆小單的跟隨率 × log(drag) × 方向 | **聲子拖曳係數**：大單後小單追隨強度，正=主力帶動散戶追漲，趨勢可能持續。 |

### H. GEO 幾何流形（2 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 22 | `f_geo_lyapunov_drift_vector` | `EWMA(ddp, span=50)/std(dp)` per 10-tick bin | **黎曼流形 Lyapunov 漂移向量**：價格第二階差分（加速度）的 EWMA，捕捉流形上的持續漂移方向。 |
| 23 | `f_geo_entropy_directed_work` | `ΣF_bin×dx_bin / (Σ|dx|×mean|F|)`, F=(BV-SV)/(BV+SV) | **熵驅動有向功**：訂單流力量（F）做的方向性功，正 = 流向與位移一致，效率高。 |

### I. NET 網路流（2 個）

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 24 | `f_net_markov_flow_asymmetry` | `(Σvol×|ΔP|_up − Σvol×|ΔP|_dn) / total` | **馬可夫鏈流非對稱**：量加權的上漲 vs 下跌位移比較，正值 = 向上流動佔優。 |
| 25 | `f_net_antiferro_broker_interaction` | Top5 買方集中度 − Top5 賣方集中度，× 買方淨流比例 | **反鐵磁性券商交互**：主要買方/賣方間的反向交互作用強度，模擬不同地盤間的反鐵磁耦合。 |

---

## 七、Alpha v8 IT Group 3 泊松通道特徵（3 個）

**輸入**：逐筆成交 (`single_stock_tick`)
**模組**：`features/single_stock_tick/`

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_it_post_event_directional_impulse` | 大單(≥90th)後10筆Tick的方向衝擊 | **大額成交事件後方向衝擊**。μ_buy_event - μ_sell_event，衡量大單後市場反應的方向性。 |
| 2 | `f_it_large_trade_iat_asymmetry` | 大單(≥90th) IAT加速度不對稱 | **大額成交到達間距不對稱**。前半段IAT / 後半段IAT，正值=賣方相對更加速。 |
| 3 | `f_it_broker_poisson_intensity_onesidedness` | 15分鐘時窗泊松強度單邊性 | **泊松強度單邊性**。(λ_buy - λ_sell) / (λ_buy + λ_sell + ε)，範圍[-1, +1]。 |

---

## 八、Alpha v9 IT Group 5 CTW 通用估計器特徵（3 個）

**輸入**：券商分點 + 逐筆成交（交叉 `cross_broker_tick`）
**模組**：`features/cross_broker_tick/`
**Batch**：20260317 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_it_ctw_log_likelihood_ratio` | 3階 Markov LLR: log(P(Y=1\|context)/P(Y=0\|context)) | **CTW對數似然比**：使用上下文樹加權法(CTW)計算3階馬爾可夫鏈的對數似然比，context=(B_{t-3},B_{t-2},B_{t-1})，捕捉券商方向歷史對價格方向的預測能力。 |
| 2 | `f_it_di_zscore` | DI滾動z分數 × ewm(net_ratio_big) | **定向資訊z分數**：1階定向資訊I(B→Y)的滾動z分數標準化後，乘以淨買比例的指數加權均值(span=5)，衡量資訊流向的時間加權強度。 |
| 3 | `f_it_markov_memory_depth_diff` | (D_dn* - D_up*) × ewm_net | **馬爾可夫記憶深度差**：找最優上漲記憶深度D_up*與最優下跌記憶深度D_dn*（分別最小化對應的log-loss），計算兩者之差乘以淨買比例ewm，捕捉多空記憶不對稱性。 |

---

## 九、Alpha v10 IT Group 6 NRDF 抽象空間特徵（4 個）

**輸入**：券商分點 + 逐筆成交（交叉 `cross_broker_tick`）
**模組**：`features/cross_broker_tick/`
**Batch**：20260317 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_it_causal_transfer_rate_deviation` | DI序列斜率 × ewm_net | **因果傳輸率偏離指數**：DI（定向資訊）在20日窗口內的線性斜率乘以當日ewm_net，正值=因果關係逐日增強。 |
| 2 | `f_it_info_weighted_price_displacement` | Σ ΔH × ΔP (30分鐘分段) | **資訊加權價格位移**：將Tick按30分鐘分成8段，計算熵降與價格變化的乘積之和，捕捉資訊流動與價格推進的同步強度。 |
| 3 | `f_it_short_long_broker_kl` | KL(q_5d ‖ q_60d) × ewm5 | **短長期 Broker 分佈 KL**：5日與60日券商淨買分佈的KL散度，乘以5日ewm淨買比例，衡量短期行為偏離長期常態的程度。 |
| 4 | `f_it_broker_distribution_hist_deviation` | KL(today ‖ 20d_med) × tanh(z) | **Broker 分佈歷史偏離極性**：今日券商分佈與20日歷史中位分佈的KL散度，乘以z分數的tanh值，捕捉分佈異常程度與方向。 |

---

## 十、Alpha v11 IT Group 7 回饋通道與守恆律特徵（3 個）

**輸入**：券商分點 + 逐筆成交（交叉 `cross_broker_tick`）
**模組**：`features/cross_broker_tick/`
**Batch**：20260317 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_it_feedback_capacity_utilization_skew` | CapUtil_up - CapUtil_dn，CapUtil = (H(Y) - H(Y|Y_{t-1})) / (H(Y) + ε) | **上行回饋利用率 skew**：上行趨勢的自強化程度與下跌趨勢的自強化程度之差，正值=上漲方向更自強化。 |
| 2 | `f_it_mi_conservation_imbalance` | (I(B;T) - I(B→T) - I(T→B)) / H(T) × delta_vwap | **MI 守恆失衡**：衡量信息架構複雜性與VWAP變化的交互作用，反映市場信息守恆律的偏離程度。 |
| 3 | `f_it_broadcast_channel_entropy_gap` | (H_small - H_large) × net_ratio_big | **廣播通道接收者熵差**：散戶與主力交易行為的熵差異乘以淨買比例，正值=散戶混亂、主力的方向明確。 |

---

## 十一、Alpha v12 生態模擬特徵（4 個）

**輸入**：逐筆成交 (`single_stock_tick`)
**模組**：`features/single_stock_tick/`
**Batch**：20260318 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_fat_tailed_seed_dispersal` | 跨越 30bps 以上的主動買單總量 / 總主動買單量 | **長尾散播基因流**：衡量大額跳價買單的分散程度，反映市場在极端价格波动时的资金扩散行为。 |
| 2 | `f_masting_predator_satiation` | 爆發分鐘內主動買入量 / 全日被動賣出量 | **Masting 掠食者撐死比**：衡量短時間突襲的力道相對於空方一整天進攻力道的強度，正值代表多方突擊強度大於空方防守。 |
| 3 | `f_mixed_levy_gaussian_dispersion` | 高斯比例 × Lévy 比例 | **Lévy-高斯混合擴散疊加態**：連續小額主動買的比例 (高斯) × 大額跨檔買的比例 (Lévy)，衡量市場在微小波動與大幅跳躍兩種狀態間的混合程度。 |
| 4 | `f_predator_mate_limitation` | plunge_sell_hhi / (plunge_buy_depth + ε) | **掠食者配偶限制**：空方大戶在急跌段（1分鐘 return < -0.3%）找不到對手盤（買方），做空效率大降，衡量空方在急跌時的集中度與深度。 |

---

## 十二、Alpha v13 生態模擬特徵（13 個）

**輸入**：券商分點 (`single_stock_broker`)
**模組**：`features/single_stock_broker/`
**Batch**：20260318 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_allee_reproduction_deficit` | (Top10買入 - 20日均值) / Bottom80%賣出，Z-score(42) | **族群繁衍臨界赤字**：大戶日買入張數的20日均值回歸程度，除以散戶賣出量，衡量主力回歸與散戶拋壓的失衡程度。 |
| 2 | `f_core_source_contribution_ratio` | 核心源區淨買 / 總淨買，滾動標準化 | **核心源區基因貢獻度**：識別主要資金來源（核心源區）對整體淨買賣的貢獻比率，反映核心主力資金的主導程度。 |
| 3 | `f_cumulative_dose_threshold` | 累積淨買超 / 閾值，Z-score標準化 | **累積劑量感染閥值**：追蹤連續買入的累積劑量，當超過歷史閾值時觸發傳染效應，捕捉市場「感染」臨界點。 |
| 4 | `f_dynamic_r0_momentum` | R0估計 × 動量因子，時間加權 | **動態R0動能**：即時估計傳染係數R0並結合動量因子，捕捉疫情式傳播動能的加速或減速。 |
| 5 | `f_herd_immunity_barrier` | (1 - 免疫比例) × 暴露率， barrier 函數標準化 | **群體免疫屏障效應**：模擬群體免疫水平形成的市場阻力，當免疫比例上升時傳染效率下降，識別市場「免疫」臨界。 |
| 6 | `f_inbreeding_depression_risk` | 相似度指標 × 劣勢表現，偏離度標準化 | **近親繁殖風險**：測量投資者群體的「基因多樣性」，高度相似性導致市場對新資訊的適應能力下降，預警極端的同質化行為。 |
| 7 | `f_intraguild_predation_skew` | 競爭者間買賣偏度，Top5 vs 次級別 | **競爭排除與同類相食**：頂級大戶與次級大戶之間的「捕食關係」，當頂級賣出而次級買入時，代表同類相食效應強化。 |
| 8 | `f_post_latency_infectivity_jump` | 潛伏期後傳染增幅 × 延遲效應 | **潛伏期後傳染跳躍**：模擬疫情傳播的潛伏期效應，識別在初始傳染後的爆發性增強，捕捉市場情緒的延遲反應。 |
| 9 | `f_primary_secondary_infection_div` | 原發感染 vs 次發感染率差異，標準化 | **原發/次發感染背離**：原發（首次買入）與次發（跟隨買入）行為的分離程度，反映市場是處於初始爆發還是跟隨擴散階段。 |
| 10 | `f_resistance_gamma_heterogeneity` | Gamma分佈異質性參數，標準化 | **抗性Gamma異質性**：使用Gamma分佈描述投資者抗性（不願買入）的異質性，高異質性代表市場抵抗力的不均衡分布。 |
| 11 | `f_stoichiometric_nutrient_imbalance` | 資本流入 / 籌碼流出 化學劑量比 | **資本-籌碼化學計量失衡**：類比化學反應中的劑量比，當資金（資本）與籌碼供應失衡時，市場將出現類似化學反應的劇烈波動。 |
| 12 | `f_tumor_fingering_breakout` | 腫瘤指狀滲透强度 × 突破閾值 | **腫瘤指狀刺透力**：模擬腫瘤細胞指狀入侵 healthy tissue的力學模型，捕捉價格向阻力位「刺透」的爆發力。 |
| 13 | `f_tumor_fingering_instability` | 指狀不穩定性指標，臨界標準化 | **橫向指狀不穩定性**：測量腫瘤指狀結構的橫向擴張不穩定性，反映市場橫向整理後的突破方向選擇不確定性。 |

---

## 十三、Alpha v14 生態模擬特徵（5 個）

**輸入**：券商分點 + 逐筆成交（交叉 `cross_broker_tick`）
**模組**：`features/cross_broker_tick/`
**Batch**：20260319 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_acellular_gap_formation` | `cumsum(spread_on_downtick × top10_net_sell_ratio)`, Z-score(42) | **空方倒貨形成的價差擴大**：空方在下跌時加大拋售力道，同時買方退縮於內盤，形成委賣價差擴大。捕捉主力倒貨時的價格微結構特徵。 |
| 2 | `f_inverse_allee_refuge_accumulation` | `Σ[Top10_BuyQtm × P_refuge(Price)] / 總量`, Z-score(42) | **主力低調吸籌（庇護所效應）**：利用5分鐘低量低波動的「庇護所時段」識別主力隱蔽建倉的行為。 |
| 3 | `f_masting_synchronous_burst` | `爆發期買入HHI集中度`, Z-score(42) | **爆發期同步行動集中度**：識別日內主動買入最集中的爆發分鐘，計算Top10在此期間買入的HHI集中度。高值代表主力同步突襲。 |
| 4 | `f_predator_driven_extinction_rate` | `Top5_Sell / PassiveBuy → pct_change(5) → zscore(42)` | **主力拋售相對於被動承接強度**：Top 5 券商賣出量相對於日內被動承接買量（PrFlag=0）的比率變化，反映主力出貨力道。 |
| 5 | `f_spatial_mixing_refuge` | `Top5 quiet_ratio - Rest quiet_ratio`, Z-score(42) | **主力避開高波動的超額吸籌**：主力在尾盤平穩期低調建倉，相對於散戶的超額吸籌比例。 |

---

## 十四、Alpha v15 生態模擬特徵（7 個）

**輸入**：券商分點 + 日頻 OHLC（交叉 `cross_broker_daily`）
**模組**：`features/cross_broker_daily/`
**Batch**：20260319 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_dilution_effect_defense` | `BuyerCount / Top10NetSellQtm (on dump day)`, EWM(10)→Z-score(42) | **生產者防禦稀釋效應**：砸盤日時淨買入券商數除以大戶淨賣出量的比例，衡量散戶/小戶在主力砸盤時的抵抗能力。 |
| 2 | `f_founder_effect_decay` | `(current_inventory / initial_buy) diff`, Z-score(42) | **創始者效應衰退**：Top5買超券商從建倉到當日累積未平倉佔最初建倉量的比例衰減程度。當此值快速下降，代表先鋒資金在撤退。 |
| 3 | `f_habitat_shifting_speed_deficit` | `speed(VWAP) - speed(Top10_VWAP)`, Z-score(42) | **棲地退移速度差**：整體VWAP下移速度與Top10大戶買入VWAP下移速度的差，捕捉主力在價格下移時的相對位置。 |
| 4 | `f_holling_type2_satiation` | `diff²(Top5Sell) × is_new_low_5d`, EWM(10)→Z-score(42) | **掠食者飽食邊界**：Top5賣出量的二階導數，只在短期創新低日計算，識別主力在高點出貨後的動能耗竭。 |
| 5 | `f_inverse_allee_exhaustion` | `(small_buyers_cnt / large_sellers_cnt) slope(10)`, Z-score(42) | **逆向阿利效應枯竭**：股價創20日新低時，逆勢買入的小型券商家數與順勢賣出的大型券商數的比率趨勢，捕捉散戶逆勢承接行為。 |
| 6 | `f_stefan_boundary_sacrifice_ratio` | `Top10_boundary_buy / Top10_total_buy`, Z-score(42) | **史蒂芬邊界獻祭比例**：大戶在創新高價位（>=daily_high*0.99）的買量佔其總買量的比例，衡量主力在高位是否願意進場。 |
| 7 | `f_super_invader_baseline_defense` | `Top10_baseline_buy / Top10_total_buy`, Z-score(42) | **超級入侵者底線防禦**：大戶在最低價區間（<=daily_low*1.01）的買量佔其總買量的比例，衡量主力在低位是否願意護盤。 |

---

## 十五、Alpha v16 生態模擬特徵（5 個）

**輸入**：逐筆成交 + 日頻 OHLC（交叉 `cross_tick_daily`）
**模組**：`features/cross_tick_daily/`
**Batch**：20260319 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_cumulative_dose_exhaustion` | `(連續上漲天數 × 期間日均量) / 當日主動賣量 → Z-score(42d)` | **累積劑量枯竭點 (買盤衰竭)**：當連續上漲天數越多且均量越大，但當日主動賣量相對小時，代表買盤即將衰竭。 |
| 2 | `f_density_dependent_dispersal_damping` | `Z-score(ModeShare) - Z-score(EscapeDist) → 再Z-score(42d)` | **密度依賴性擴散阻尼**：ModeShare 高代表高密度，EscapeDist 小代表走不動。兩者之差捕捉價格在高密度區間的擴散阻力。 |
| 3 | `f_double_threshold_nonlinear_surge` | `同時跨越VWAP與開盤價的主動買量 / 總主動買量 → Z-score(42d)` | **雙重閾值非線性躍變**：同時跨越VWAP與開盤價兩個閾值的主動買單量佔比，代表價格突破多個心理關卡的力度。 |
| 4 | `f_limit_cycle_exhaustion_trap` | `log((VWAP穿越次數 × 總成交量) / 價格位移幅度) → Z-score(42d)` | **動力學極限環耗散枯竭**：VWAP 穿越次數多但價格位移小，代表市場在高頻來回震盪但無有效趨勢，是動能耗散的陷阱區。 |
| 5 | `f_shifting_habitat_patch_deficit` | `推升速度 / (Mode±0.5%成交量佔比 + ε) → Z-score(42d)` | **斑塊存活赤字**：價格重心推升過快但最大成交量區間極度單薄，代表市場在缺乏流動性支撐下快速位移，是暴漲暴跌的前兆。 |

**共同前置處理** (`preprocess_cross_tick_daily`):
- 重用 `preprocess_single_stock_tick` 取得 tick 中間變數 (vwap, buy_vol, sell_vol, total_vol)
- 合併 daily OHLC 資料 (收盤價, 開盤價)
- 計算 cross 中間變數: raw_consecutive_up_days, raw_consecutive_up_vol, raw_mode_share, raw_mode_price, raw_patch_thickness, raw_push_speed, raw_cross_count, raw_cross_vol, raw_cross_both_vol, raw_total_active_buy, raw_active_sell
- 新增 intermediates: raw_prior_pessimism, raw_ignorant_selling, raw_total_vol_50bin, raw_neg_mood, raw_mood_sell, raw_gap_down, raw_instant_exit, raw_loss_domain_max, raw_high_prior, raw_bin1_large_buy

---

## 二十一、Alpha v17 行為經濟學特徵（交叉逐筆日頻，4 個）

**輸入**：逐筆成交 + 日頻 OHLC（交叉 `cross_tick_daily`）
**模組**：`features/cross_tick_daily/`
**Batch**：20260327 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_be_prior_bias_neglect` | `前日跳空低開(開盤<前日收盤*0.99) × 反彈區間的小單賣出額 / 總成交量` → rolling_zscore(20) | **前日偏見忽視**：前日跳空低開代表市場過度悲觀，在反彈區間（Bin收盤>開盤）的小單賣出代表散戶過度反應。捕捉市場對前日訊號的反向修正能力。 |
| 2 | `f_be_mood_congruent_action` | `前日重挫跳空(開盤<前日收盤*0.97) × 上漲區間的小單賣出額` → rolling_zscore(20) | **情緒一致性行動**：前日大跌且跳空低開代表空頭情緒蔓延，但在上漲區間（Bin收盤>開盤）的小單賣出代表散戶在反彈中過早退出。捕捉散戶情緒與價格行為的偏離。 |
| 3 | `f_be_hazard_rate_capitulation` | `開盤跳空低開(開盤<前日收盤*0.98) × 前5個Bin的小單賣出總額` → ewm_then_zscore(5, 20) | **投降率衰竭**：開盤跳空低開代表市場開盤即確認跌勢，前5個Bin的小單賣出總額代表散戶在開盤後的持續拋售行為。捕捉市場在下跌趨勢中的崩潰程度。 |
| 4 | `f_be_loss_domain_risk_seeking` | `處於虧損區間(VWAP<開盤*0.98) × (小單成交價標準差/均價)` 的日最大值 → ewm_then_zscore(5, 20) | **虧損區間風險尋求**：VWAP < 開盤*0.98 代表當日處於虧損區間，小單成交價的標準差/均價代表散戶在虧損時的恐慌程度。捕捉散戶在虧損狀態下的非理性行為。 |

---

## 二十二、Alpha v17 遊戲理論特徵（交叉逐筆日頻，1 個）

**輸入**：逐筆成交 + 日頻 OHLC（交叉 `cross_tick_daily`）
**模組**：`features/cross_tick_daily/`
**Batch**：20260327 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_binary_trust_leap` | `近20日漲幅>15% × (開盤首個Bin的大單買入/20日均值)` → rolling_zscore(20) | **二進制信任跳躍**：當股票近20日漲幅超過15%時，代表處於強勁上漲趨勢。開盤第一個Bin的大單買入代表主力在趨勢確認後的積極建倉行為。捕捉主力在趨勢確認後的信心程度。 |

---

## 十六、Alpha v17 遊戲理論特徵（9 個）

**輸入**：逐筆成交資料（`trade_level1_data`）
**模組**：`features/single_stock_tick/`
**Batch**：20260322 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_cheap_talk_distraction_trap` | 1-lot buys (PrFlag=1, DealCount=1) 為噪音掩護，大額賣出 (DealCount≥q95) 為實際行動。Large sell vol / 1-lot buy count → Z-score(20d) | **廉價話語分心陷阱**：1 手散單掩護大戶真實派發意圖。比率越高代表大戶出貨越隱蔽。 |
| 2 | `f_gt_feigned_ignorance_accumulation` | Inner market (PrFlag=0): Small active sell (DealCount≤q50) 為假象，Large passive buy (DealCount≥q95) 為真實吸籌。sas_count × lpb_vol → Z-score(20d) | **裝傻吸籌陷阱**：小戶主動賣出掩護大戶被動買入。乘積越大代表大戶低調吸籌越積極。 |
| 3 | `f_gt_feigned_ignorance_distribution` | Outer market (PrFlag=1): Small active buy (DealCount≤q50) 拉升價格，Large passive sell (DealCount≥q95) 派發。sab_count × lps_vol → Z-score(20d) | **裝傻派發陷阱**：小戶主動買入吸引散戶追高，大戶趁機出貨。乘積越大代表主力派發越成功。 |
| 4 | `f_gt_gradual_fleeing_capitulation` | 500-tick bins: 連續 4 個 bin 的 active buy (PrFlag=1) 遞減，代表聰明錢放棄。sell_surge / sell_ma → EWM(3) → Z-score(20d) | **漸進逃離投降**：聰明錢連續拋售後的最後一波投降式賣出。信號越強代表市場越恐慌。 |
| 5 | `f_gt_invidious_distinction_cost` | 500-tick bins: Large active buy (PrFlag=1, DealCount≥q95) 的 VWAP 溢價 × 成交量。(large_buy_vwap / bin_vwap - 1) × lb_vol → Z-score(20d) | **嫉妒代價**：大戶願意高於市場均價買入，顯示其資訊優勢。溢價越高代表大戶信心越強。 |
| 6 | `f_gt_pecuniary_emulation_trap` | 小戶模仿昨日大戶方向 (sign_mimic)，但大戶已反轉 (sign_reversal)。is_mimic × is_reversal × |large_netflow| → Z-score(20d) | **金錢模仿陷阱**：散戶模仿落後信號而大戶已反轉。代表散戶被套牢的風險。 |
| 7 | `f_gt_persuasion_suspense_slide` | 100-tick bins: 沉默偵測 (bin_vol < 0.5 × prev_20_bin_ma)，沉默連續後的大額主動買入。silence_streak × large_buy_vol → Z-score(20d) | **說服懸念滑落**：長期沉默後的突然大額買入伏擊。代表主力暗中吸籌後的爆發。 |
| 8 | `f_gt_uninformed_herding_bias` | 100-tick bins: 小單 (DealCount≤q50) 跟隨前期價格方向的比例。follow_rate → Z-score(20d) | **無知羊群偏誤**：散戶盲目跟隨前期價格趨勢而忽視基本面。跟隨率越高代表市場越是非理性。 |
| 9 | `f_gt_forward_induction_signal` | AM (前 1000 ticks) 安靜 vs PM (後 1000 ticks) 積極大額買入 (DealCount≥q95)。(PM_Large_Buy / AM_Large_Buy) × is_am_quiet → Z-score(20d) | **前向歸納信號**：大戶在早盤保持沉默，午後積極買入。前向歸納邏輯隱藏其真實意圖。 |

**閾值計算**：
- q95: 滾動 20 日均值，shift(1) 避免未來資料
- q50: 滾動 20 日均值，shift(1) 避免未來資料
- Z-score: (x - rolling_mean) / (rolling_std + ε)，min_periods=max(1, window//2)

**共同前置處理** (`single_stock_tick`):
- 資料來源：`trade_level1_data` (逐筆成交)
- 排序：按 `TotalQty` (非 `DealTimeSecond`) 確保同秒內訂單的確定性排序
- 輸出格式：`(StockId, Date)` 為唯一鍵

---

## 十七、Alpha v17 遊戲理論特徵（日頻，6 個）

**輸入**：日頻 OHLCV + 籌碼（`daily_data_bystock`）
**模組**：`features/single_stock_daily/`
**Batch**：20260322 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_level_k_retail_harvesting` | `corr(當沖%, ret5) × (-sign(inst_net)) × |inst_net|` → Z-score(20d) | **Level-k 散戶收割**：當沖熱度與報酬正相關但法人反向操作。法人借散戶慣性收割。 |
| 2 | `f_gt_unraveling_skepticism_failure` | `is_bad(rev_yoy≤0 & ret5≤0) × retail_growth_5` → EWM(3) | **解讀失敗懷疑**：散戶在營收衰退且價格下跌時仍增加持股。代表散戶過度自信。 |
| 3 | `f_gt_delayed_disclosure_moral_hazard` | `1/vol_20d × short_change_5d` → Z-score(60d) | **延遲揭露道德危機**：低波動假象掩護融券餘額上升。代表內部人借券布局。 |
| 4 | `f_gt_reputation_cost_signaling` | `down_3d × inst_change × is_net_positive` → Z-score(20d) | **名譽成本信號**：法人在連續下跌後強力買入。顯示其對基本面更有信心。 |
| 5 | `f_gt_strategic_ignorance_demand` | `is_below_ma × is_falling × is_margin_up × margin_change_3` → Z-score(60d) | **戰略性無知需求**：散戶在跌破均線且下跌5%時仍增加融資。代表拒絕承認錯誤。 |
| 6 | `f_gt_moral_hazard_leverage` | `(dt_pct + margin_usage) / large_holder_pct` → Z-score(60d) | **道德危機槓桿**：散戶當沖+融資同步放大但大戶退出。槓桿錯配最危險情境。 |

**共同前置處理** (`preprocess_single_stock_daily`):
- 計算 intermediate columns: `_ret_5d`, `_inst_net`, `_inst_net_change`, `_short_total`, `_short_change_5d`, `_vol_20d`, `_sma_20`, `_margin_bal`, `_margin_change_3`, `_margin_usage`, `_large_holder_pct`, `_rev_yoy`, `_retail_count`
- 輸出格式：`(StockId, Date)` 為唯一鍵

---

## 十八、Alpha v18 行為經濟學特徵（日頻，4 個）

**輸入**：日頻 OHLCV + 籌碼（`daily_data_bystock`）
**模組**：`features/single_stock_daily/`
**Batch**：20260328 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_be_asymmetric_confidence_spillover` | `DT_Trend × Bad_News` → Z-score(20d) | **不對稱自信溢出**：當沖比例相對於20日均值的趨勢 × 3日報酬<-3%壞消息。捕捉散戶過度自信，在市場下跌時仍積極參與。 |
| 2 | `f_gt_outgroup_exploitation_bias` | `Outgroup_Influx × Inst_Exploitation` → EWM(5) → Z-score(42d) | **外團體剝削偏誤**：散戶人數增長率 × 法人淨賣出。捕捉散戶湧入時法人倒貨坑殺外團體的行為。 |
| 3 | `f_be_ego_driven_misreporting` | `trend_down × margin_increase × margin_diff / margin_bal` → Z-score(20d) | **自我驅動錯誤報告**：股價跌破SMA20且5日報酬<-3%時，融資餘額增加。捕捉散戶因自我 ego 而持續加碼攤平的行為。 |
| 4 | `f_be_frustration_driven_turnover` | `high_expectation × disappointment × turnover_rate` → Z-score(20d) | **挫折驅動換手**：股價突破20日新高但收盤翻黑 × 換手率。捕捉投資人挫折情緒引發的爆量換手。 |

**共同前置處理** (`preprocess_single_stock_daily`):
- 新增 intermediate columns: `_ret_3d`, `_dt_mean_20`
- 輸出格式：`(StockId, Date)` 為唯一鍵

---

## 十八、Alpha v17 遊戲理論特徵（券商分點，3 個）

**輸入**：券商分點 (`single_stock_broker`)
**模組**：`features/single_stock_broker/`
**Batch**：20260322 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_payoff_dominant_focal_point` | 聚焦價格（10的倍數）的不同券商數量 × 買入量 → EWM(5) → Z-score(20) | **聚焦點 payoff dominant**：多個券商聚焦於整數價位形成支撐。當聚焦點共識越強，代表市場對該價位有強烈共識，後續價格可能向該方向突破。 |
| 2 | `f_gt_expert_vs_popular_divergence` | Top-5 成交量券商淨買 > 0 AND Mid-10 券商淨賣 > 0 → 信號 × 差距 → EWM(5) → Z-score(20)（使用日級別聚合近似） | **專家 vs 大眾背離**：當高成交量券商（top-5 by buy volume）淨買但中階券商（rank 6-15）淨賣時，代表大戶正在派發，後續價格可能下跌。（原始版本需 Rolling per-broker metrics，本版本使用日級別近似） |
| 3 | `f_gt_game_of_chicken_standoff` | Buy HHI 和 Sell HHI 都處於60日90百分位且差異很小 → 1/差異 × EWM(3) | **膽小鬼僵局**：買賣雙方都高度集中但勢均力敵。當雙方都認為自己會贏但都不願先讓步時市場將出現高波動僵局。 |

**共同前置處理** (`preprocess_single_stock_broker`):
- 資料來源：`broker_data` (券商分點進出資料)
- 新增 intermediate columns: `raw_focal_distinct_brokers`, `raw_focal_buy_vol`, `raw_hhi_buy`, `raw_hhi_sell`
- 輸出格式：`(StockId, Date)` 為唯一鍵

---

## 十九、Alpha v17 遊戲理論特徵（交叉券商日頻，8 個）

**輸入**：券商分點 + 日頻 OHLC（交叉 `cross_broker_daily`）
**模組**：`features/cross_broker_daily/`
**Batch**：20260322 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_costly_waiting_attrition` | `rolling_sum(Top5_NetBuy, 5) × is_falling(跌5日報酬<0)` → Z-score(20d) | **消耗戰鎖碼**：股價連跌時前5大主力仍持續淨買，滾動5日加總×是否下跌作為 raw，zscore(20)標準化。遊戲理論中展示承受虧損的決心。方向：正向 → 大戶在跌勢中仍堅持 → 後續可能反轉。 |
| 2 | `f_gt_echo_chamber_assortative` | `vol_surge × (1 - top10_ratio)` → EWM(5) → Z-score(20d) | **同溫層泡泡**：成交量放大但前10大主力參與率下降，代表散戶對敲泡沫。ewm(5)→zscore(20)標準化。方向：正向 → 散戶對敲越活絡 → 極端可能反轉。 |
| 3 | `f_gt_second_order_belief_fading` | `is_breakout(收盤>20日高) × Top5_NetSell` → Z-score(60d) | **二階信念褪散**：股價突破20日新高時，前5大主力趁機淨賣出收割。is_breakout×Top5_NetSell作 raw，zscore(60)標準化。方向：正向 → 假突破陷阱 → 極端低報酬。 |
| 4 | `f_gt_commitment_device_lock` | `is_strong_up(5日報酬>5%) × is_minimal_sell(賣出量<=20日5%分位)` → EWM(3) | **承諾機制鎖碼**：近期大漲(5日>5%)且過去20日累積買超最大之前10大券商今日賣出量為0或極低。ewm(3)標準化。方向：正向 → 絕對鎖碼決心 → 極端高報酬。 |
| 5 | `f_gt_gradual_trust_escalation` | `rolling_zscore(trust_test × escalation × price_ok, 20)` | **漸進信任升級（消耗戰鎖碼）**：Top5主力持續淨買≥3日且價格支撐完好時，信號增強。代表主力願意承受短期虧損以達長期目標。trust_test: Top5淨買≥3日; escalation: 當期/前期Top5淨買; price_ok: 3日報酬>-1%。方向：正向 → 主力持續累計 → 後續可能反轉。 |
| 6 | `f_be_overconfidence_spillover` | `rolling_zscore(condition × top5_5d_netbuy, 20)` | **過度自信溢價**：過去5日報酬>5%且今日開盤大跌>2%時，前5主力仍未減碼。代表主力過度自信而忽略風險。condition: past_success(5日報酬>5%) AND today_shock(開盤<0.98×前收)。方向：正向 → 主力過度自信 → 極端可能反轉。 |
| 7 | `f_gt_pooling_equilibrium_camouflage` | `ewm_then_zscore(vol_surge × low_concentration, 5, 20)` | **Pool均衡偽裝**：成交量放大但集中度低，代表散戶對敲掩護主力進出的偽裝行為。vol_surge: volume>1.5×20日均量; low_concentration: HHI_buy<20日分位數。方向：正向 → 散戶對敲活絡 → 極端可能反轉。 |
| 8 | `f_gt_incumbent_signaling_exemption` | `ewm_then_zscore(1 / (abs_ret / passive_proxy + ε), 5, 20)` | **現任者豁免**：主力被動買入強度高但價格波動小，代表無需拉抬即可鎖碼的現任者優勢。passive_proxy: incumbent_buy / rolling_mean(incumbent_buy, 10); abs_ret: \|收盤-開盤\| / 開盤。方向：正向 → 現任者鎖碼穩固 → 支撐強勁。 |

---

## 二十、Alpha v18 賽局理論/行為經濟學特徵（交叉券商日頻，10 個）

**輸入**：券商分點 + 日頻 OHLC（交叉 `cross_broker_daily`）
**模組**：`features/cross_broker_daily/`
**Batch**：20260325 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_risk_excuse_dumping` | `top5_net_sell × high_risk` → Z-score(20d) | **風險藉口倒貨**：當股價波動大時，前5大主力券商持續賣出。使用滾動60日計算主力券商，high_risk = (最高價-最低價)/開盤價。方向：正向 → 主力借波動出貨 → 後續可能下跌。 |
| 2 | `f_gt_partial_lying_stealth_sell` | `rank_3_5_sell / top1_sell` → Z-score(20d) | **部分說謊隱蔽賣出**：隱藏最大賣家的同時，其他賣家持續出貨。計算排名第3-5名賣出量之和與第1名之比。方向：正向 → 隱蔽派發 → 極端可能反轉。 |
| 3 | `f_gt_in_group_stealth_collusion` | `collusion × intensity` → Z-score(20d) | **內團體隱蔽合謀**：前5大券商與其他券商方向相反。collusion = 1 if top5_net > 0 and rest_net < 0。方向：正向 → 主力合謀 → 價格操縱。 |
| 4 | `f_gt_heterogeneous_stigma_blending` | `inst_dump × blending` → EWM(5) → Z-score(20d) | **污名混合分散**：賣方集中度低且賣出量大時。blending = 1/HHI_sell，inst_dump = 1 if sell > mean*1.5。方向：正向 → 散戶恐慌拋售 → 極端可能反轉。 |
| 5 | `f_gt_cooperation_breakdown_threshold` | `beyond_threshold × top5_defection` → Z-score(20d) | **合作崩潰閾值**：買方券商數激增但主力券商叛離。beyond_threshold = 1 if buy_group_size > rolling_max(20)。方向：正向 → 主力叛離 → 後續下跌。 |
| 6 | `f_be_in_group_favoritism_cross` | `cohesion × outgroup_ns` → EWM(5) → Z-score(20d) | **內團體偏愛交叉**：主力買方一致性高且外部賣壓大。cohesion = 1/(CV+ε)，outgroup_ns = rest_sell - rest_buy。方向：正向 → 主力護盤但散戶跑 → 支撐股價。 |
| 7 | `f_gt_character_stigma_avoidance` | `heavy_seller / stigma_gap` → EWM(5) → Z-score(20d) | **品格污名迴避**：頂級賣家出貨但掩蓋痕跡。stigma_gap = (r1-r2)+(r1-r3)，heavy = 1 if r1 > mean*2。方向：正向 → 刻意隱藏賣出 → 後續可能下跌。 |
| 8 | `f_be_status_modulated_denial` | `gap_down × top5_ns × rest_nb_norm` → EWM(5) → Z-score(20d) | **地位調節否認**：跳空下跌時主力隱瞞賣出。gap_down = 1 if open < prev_close*0.98。方向：正向 → 主力隱瞞賣出 → 持續下跌。 |
| 9 | `f_be_antisocial_exit_inequality` | `inequality × antisocial_exit` → Z-score(20d) | **反社會退出不平等**：買方集中度高但參與者減少。inequality = 1 if top1_ratio > q90。方向：正向 → 大戶壟斷買盤 → 流動性枯竭。 |
| 10 | `f_gt_costly_signaling_exclusion` | `max(0, premium) × group_strength` → Z-score(20d) | **高昂信號排他鎖碼**：主力高價買入鎖碼。premium = top3_avg_price/daily_vwap - 1。方向：正向 → 主力高價鎖碼 → 後續上漲。 |

**共同前置處理** (`preprocess_cross_broker_daily`):
- 資料來源：`broker_data` (券商分點) + `daily_data_bystock` (日頻 OHLC)
- 新增 intermediate columns: `raw_top5_net_buy`, `raw_top5_net_sell_qtm`, `raw_top10_total_buy`
- 輸出格式：`(StockId, Date)` 為唯一鍵

---

## 二十、Alpha v17 遊戲理論特徵（交叉逐筆日頻，1 個）

**輸入**：逐筆成交 + 日頻 OHLC（交叉 `cross_tick_daily`）
**模組**：`features/cross_tick_daily/`
**Batch**：20260323 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_immediate_disclosure_panic` | `gap_down(開盤價 < 前日收盤 × 0.98) × first_500_tick_sell_ratio` → Z-score(20d) | **立即揭露恐慌**：開盤跳空下跌 ≥ 2% 且前 500 筆買賣比賣壓過重，代表大戶對利空訊息的暴力拋售。正值 → 極端低報酬。 |

**閾值計算**：
- gap_down: 使用前一日收盤價 `shift(1)`，確保不使用未來資料
- first_500_tick_sell_ratio: 前 500 筆成交中，主動賣出（PrFlag=0）張數 / 總張數
- Z-score: (x - rolling_mean) / (rolling_std + ε)，min_periods=max(1, window//2)

**共同前置處理** (`preprocess_cross_tick_daily`):
- 資料來源：`trade_level1_data` (逐筆成交) + `daily_data_bystock` (日頻 OHLC)
- 重用 `preprocess_single_stock_tick` 取得 tick 中間變數
- 合併 daily OHLC 資料 (收盤價, 開盤價)
- 輸出格式：`(StockId, Date)` 為唯一鍵

---

## 二十七、AFD 大氣流體力學特徵（新增，交叉逐筆日頻，10 個）

**輸入**：逐筆成交資料 + 日頻 OHLC 資料
**模組**：`features/cross_tick_daily/calculate_f_afd_*.py`

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_afd_cold_sector_convective_trigger` | `cold_zone_buy × cold_slope` → Z-score(20d) | **冷區對流加熱**：冷區（均價以下）主動買量 × 冷區價格斜率，代表冷區資金進場推動價格上漲的對流效應。 |
| 2 | `f_afd_latent_heat_waterfall_slope` | `融券餘額 × tick_slope` (當 both > 0) → Z-score(42d) | **融券槓桿點火**：融券餘額與正向價格斜率同時存在，代表空頭回補燃料點燃上漲動能。 |
| 3 | `f_afd_baroclinic_ageostrophic_div` | `max(0, close_p - vwap) × total_vol` → log1p → Z-score(20d) | **前沿價格脫離重心**：價格偏離日內均值的程度 × 成交量，代表脫離均衡態的發散強度。 |
| 4 | `f_afd_pv_pt_resonance_jacobian` | `det(vwap_slope × tick_slope) × max(0, close_p - vwap)` → Z-score(42d) | **宏觀微觀斜率共振**：VWAP 宏觀趨勢與 tick 微觀趨勢的 Jacobian 行列式，捕捉兩尺度共振。 |
| 5 | `f_afd_storm_relative_helicity` | `(tick_mean_price - vwap) × 5日報酬滾動和` → Z-score(42d) | **風暴相對螺旋度**：價格相對重心與環境報酬的外積，捕捉價格旋轉動能。 |
| 6 | `f_afd_helical_turbulence_filter` | `(large_buy / (small_buy + 1)) × max(0, close_p - open_p)` → log1p → Z-score(42d) | **高螺旋度濾波**：大單/小單比率 × 正向價格變動，代表機構動能 filter 過濾雜訊。 |
| 7 | `f_afd_occlusion_zone_vorticity` | `(above_vwap_buy - above_vwap_high_buy) × (close_p - open_p)` → Z-score(20d) | **錮囚區渦度**：廣大買盤支撐與高價區賣壓的渦度，代表多空交戰區的旋轉。 |
| 8 | `f_afd_ground_relative_friction_shear` | `market_ret × (stock_ret - market_ret) × vol` → Z-score(20d) | **大盤摩擦剪切**：個股相對大盤的超額報酬 × 成交量，捕捉相對摩擦效應。 |
| 9 | `f_afd_optimal_drag_coefficient` | `Gaussian_kernel(drag_ratio) × max(0, slope)` → Z-score(42d) | **最佳阻力係數**：基於 20 日中位數的高斯核權重，篩選最優價格阻力位置。 |
| 10 | `f_afd_friction_riverbend_exchange` | `spread_shift × net_buy × positive_deltaP` (50-tick bins) → Z-score(20d) | **河彎效應**：價格流過彎道（VWAP 交叉）的交換動力學，代表流體摩擦耗散。 |

**共同前置處理** (`preprocess_cross_tick_daily` + `_compute_afd_intermediates`):
- 資料來源：`trade_level1_data` (逐筆成交) + `daily_data_bystock` (日頻 OHLC)
- AFD 中間變數：`raw_cold_zone_buy`, `raw_cold_slope`, `raw_tick_mean_price`, `above_vwap_buy`, `above_vwap_high_buy`, `tick_slope`, `raw_margin_balance`, `raw_drag_ratio`, `raw_riverbend_val`
- 重用 `preprocess_single_stock_tick` 取得基礎 tick 中間變數
- 合併 daily OHLC 資料 (收盤價, 開盤價, 報酬率, 融券餘額)
- 輸出格式：`(StockId, Date)` 為唯一鍵

---

## 概念分類索引

### 生態模擬 / 流行病學
`f_fat_tailed_seed_dispersal`, `f_masting_predator_satiation`, `f_mixed_levy_gaussian_dispersion`, `f_predator_mate_limitation`, `f_allee_reproduction_deficit`, `f_core_source_contribution_ratio`, `f_cumulative_dose_threshold`, `f_dynamic_r0_momentum`, `f_herd_immunity_barrier`, `f_inbreeding_depression_risk`, `f_intraguild_predation_skew`, `f_post_latency_infectivity_jump`, `f_primary_secondary_infection_div`, `f_resistance_gamma_heterogeneity`, `f_stoichiometric_nutrient_imbalance`, `f_tumor_fingering_breakout`, `f_tumor_fingering_instability`, `f_acellular_gap_formation`, `f_inverse_allee_refuge_accumulation`, `f_masting_synchronous_burst`, `f_predator_driven_extinction_rate`, `f_spatial_mixing_refuge`, `f_dilution_effect_defense`, `f_founder_effect_decay`, `f_habitat_shifting_speed_deficit`, `f_holling_type2_satiation`, `f_inverse_allee_exhaustion`, `f_stefan_boundary_sacrifice_ratio`, `f_super_invader_baseline_defense`

### 遊戲理論 / 資訊經濟學
`f_gt_cheap_talk_distraction_trap`, `f_gt_feigned_ignorance_accumulation`, `f_gt_feigned_ignorance_distribution`, `f_gt_gradual_fleeing_capitulation`, `f_gt_invidious_distinction_cost`, `f_gt_pecuniary_emulation_trap`, `f_gt_persuasion_suspense_slide`, `f_gt_uninformed_herding_bias`, `f_gt_forward_induction_signal`, `f_gt_immediate_disclosure_panic`, `f_gt_level_k_retail_harvesting`, `f_gt_unraveling_skepticism_failure`, `f_gt_delayed_disclosure_moral_hazard`, `f_gt_reputation_cost_signaling`, `f_gt_strategic_ignorance_demand`, `f_gt_moral_hazard_leverage`, `f_gt_payoff_dominant_focal_point`, `f_gt_expert_vs_popular_divergence`, `f_gt_game_of_chicken_standoff`, `f_gt_costly_waiting_attrition`, `f_gt_echo_chamber_assortative`, `f_gt_second_order_belief_fading`, `f_gt_commitment_device_lock`, `f_gt_risk_excuse_dumping`, `f_gt_partial_lying_stealth_sell`, `f_gt_in_group_stealth_collusion`, `f_gt_heterogeneous_stigma_blending`, `f_gt_cooperation_breakdown_threshold`, `f_gt_character_stigma_avoidance`, `f_gt_costly_signaling_exclusion`

### 行為經濟學
`f_be_in_group_favoritism_cross`, `f_be_status_modulated_denial`, `f_be_antisocial_exit_inequality`, `f_be_salience_misallocation`, `f_be_safe_environment_exit_spite`, `f_be_task_confidence_contagion`, `f_be_prior_biased_garbling`, `f_be_asymmetric_confidence_spillover`, `f_be_ego_driven_misreporting`, `f_be_frustration_driven_turnover`, `f_gt_outgroup_exploitation_bias`

### 大氣流體力學 (AFD)
`f_afd_cold_sector_convective_trigger`, `f_afd_latent_heat_waterfall_slope`, `f_afd_baroclinic_ageostrophic_div`, `f_afd_pv_pt_resonance_jacobian`, `f_afd_storm_relative_helicity`, `f_afd_helical_turbulence_filter`, `f_afd_occlusion_zone_vorticity`, `f_afd_ground_relative_friction_shear`, `f_afd_optimal_drag_coefficient`, `f_afd_friction_riverbend_exchange`

---

## 二十三、Alpha v17 行為經濟學特徵（新增，交叉逐筆日頻，5 個）

**輸入**：逐筆成交 + 日頻 OHLC（交叉 `cross_tick_daily`）
**模組**：`features/cross_tick_daily/`
**Batch**：20260327 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_be_salience_misallocation` | `salience/vol_quality 日內最大值` → ewm_then_zscore(5, 20) | **顯著性誤配**：價格區間寬度相對於成交量品質的比值，捕捉價格波動與成交密度的異常配對。 |
| 2 | `f_be_safe_environment_exit_spite` | `safe_env × failed_breakout × (pm_small_sell / total_vol)` → rolling_zscore(20) | **安全環境離場怨恨**：低波動環境中的失敗突破搭配尾盤小單賣出，捕捉市場在觀望後的失望拋售。 |
| 3 | `f_be_task_confidence_contagion` | `yesterday_success × gap_down × am_retail_buy` → rolling_zscore(20) | **任務信心傳染**：昨日成功搭配跳空低開，早盤小單買入代表散戶試圖抄底，捕捉市場逆勢承接行為。 |
| 4 | `f_be_prior_biased_garbling` | `bull_prior × garbling × dumping` → rolling_zscore(20) | **事前偏見混淆**：強勢市場中的買賣不平衡與價格下跌搭配，捕捉主力派發前的混淆訊號。 |
| 5 | `f_gt_costly_screening_premium` | `is_costly × premium × (bin1_lb / total_lb)` → rolling_zscore(20) | **代價篩選溢價**：高於前高開盤溢價搭配首檔大單主導度，捕捉主力刻意高開試盤行為。 |

---

## 二十四、Alpha v17 遊戲理論特徵（新增，交叉逐筆日頻，2 個）

**輸入**：逐筆成交 + 日頻 OHLC（交叉 `cross_tick_daily`）
**模組**：`features/cross_tick_daily/`
**Batch**：20260327 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_excuse_driven_risk_taking` | `loss_domain × volatility × sell_vol 日最大值` → rolling_zscore(20) | **藉口驅動風險承担**：虧損區間的波動性與賣出量結合，捕捉散戶在虧損時的冒險行為。 |
| 2 | `f_gt_reputation_manipulation_fraud` | `small_buy_ticks intensity × top5_net_sell` → rolling_zscore(20) | **名譽操作欺詐**：小額買單噪音強度與主力淨賣出金額的乘積，捕捉主力刻意製造活躍假象掩護出貨。 |

---

## 二十五、Alpha v17 遊戲理論特徵（新增，交叉券商日頻，1 個）

**輸入**：券商分點 + 日頻 OHLC（交叉 `cross_broker_daily`）
**模組**：`features/cross_broker_daily/`
**Batch**：20260327 新增

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_gt_reputation_manipulation_fraud` | `small_buy_ticks × top5_net_sell` → rolling_zscore(20) | **名譽操作欺詐**：結合券商淨賣出與小額買單噪音強度，捕捉主力利用散單掩護真實派發意圖。 |

---

## 二十六、AFD 大氣流體力學特徵（24 個）

**輸入**：逐筆成交 (`single_stock_tick`)
**模組**：`features/single_stock_tick/`
**Batch**：20260330 新增

**重要說明**：AFD 特徵計算出的 raw 值極小（~1e-9 到 1e-12），在使用 `zscore_rolling` 時需要較小的 epsilon 以避免數值穩定性問題。所有 AFD 特徵使用 `eps=1e-10` 參數呼叫 `zscore_rolling`，與原始測試程式碼一致。

| # | 特徵名 | 計算方式 | 概念邏輯 |
|---|--------|---------|---------|
| 1 | `f_afd_bifurcation_tipping_point` | 200-tick bin能量閾值與方向加權 | **臨界分岔點**：系統能量突變時的方向性觸發，捕捉價格動能的臨界加速效應。 |
| 2 | `f_afd_bred_vector_shear_penalty` | 60/40 split 內積與外積幾何交叉 | **繁殖向量剪切 penalty**：前段趨勢與後段擾動的幾何交叉，捕捉洗盤後轉折。 |
| 3 | `f_afd_condensation_burst_asymmetry` | 突破賣一價的凝結觸發與三次方放大 | **凝結爆發非對稱**：三次方運算子放大向上掃盤的極端非對稱方向。 |
| 4 | `f_afd_continuous_spectrum_transient` | 1-tick與60-tick趨勢對齊 | **連續譜暫態**：超高頻與長週期趨勢的瞬間對齊，捕捉非模態增長。 |
| 5 | `f_afd_crosswise_to_streamwise` | 掛單不平衡×主動買量×正向價格 | **橫向到流向交換**：掛單不平衡與主動買量的交互作用，捕捉河彎效應。 |
| 6 | `f_afd_diabatic_conversion_cross` | V×ΔP 正向與全方向轉換差異 | **非絕熱轉換交叉**：主動單方向與價格位移的正交性，區分真假突破。 |
| 7 | `f_afd_latent_heat_phase_locking` | 30-tick bin 內買賣序列內積 | **潛熱相位鎖定**：買賣單在時間序列上的瞬時同步性，捕捉賣單被秒吞噬。 |
| 8 | `f_afd_lc1_frontal_gravity_wave` | 掛單壓力與價格位移外積 | **LC1 前鋒重力波**：掛單壓力與價格位移的正交程度，捕捉冷鋒激發效應。 |
| 9 | `f_afd_lc2_cyclonic_breaking` | 100-tick bin 買壓制賣動能 | **LC2 氣旋碎波**：買盤壓制賣盤的動能累積，捕捉強烈氣旋不對稱性。 |
| 10 | `f_afd_lfcb_convergence_tensor` | 買賣推進梯度的張量跡 | **邊界水平輻合張量**：買賣推進梯度的張量跡，捕捉垂直渦度拉伸。 |
| 11 | `f_afd_moist_baroclinic_acceleration` | 5-tick與60-tick斜率差×ΔP | **濕斜壓加速度**：凝結潛熱使線性成長轉化為指數級爆發，捕捉成長率二階加速度。 |
| 12 | `f_afd_moist_enstrophy_jacobian` | 50-tick bin Jacobian 行列式 | **濕度擬能 Jacobian**：Jacobian 行列式衡量相空間收縮/擴張，捕捉非線性演化。 |
| 13 | `f_afd_nlsv_meridional_elongation` | q85/q50 大小單對齊度 | **NLSV 經向拉長**：大小單向量的對齊度，捕捉非線性拉伸效應。 |
| 14 | `f_afd_nlsv_meridional_extension_tensor` | 時空域張量跡 | **NLSV 經向延展張量**：時空域張量跡，捕捉波干擾防止效應。 |
| 15 | `f_afd_nlsv_zonal_shear_alignment` | 大小單同向向上瞬間協同 | **NLSV 帶狀剪切對齊**：大小單同向向上的瞬間協同，捕捉帶狀剪切效應。 |
| 16 | `f_afd_optimal_perturbation_jacobian` | 2×2 斜率矩陣行列式 | **最佳擾動 Jacobian**：2×2 斜率矩陣的行列式，捕捉最大能量增幅。 |
| 17 | `f_afd_pv_unshielding_orr` | 掛單結構與市價成交正交度 | **位渦去屏蔽**：掛單結構與市價成交的正交度，捕捉奧爾效應。 |
| 18 | `f_afd_streamfunction_variance_filter` | 大單信號/碎單噪聲比 | **流函數方差濾波**：大單正向推升與碎單噪聲的信雜比。 |
| 19 | `f_afd_sv_optimization_time_divergence` | EWM short vs long 發散度 | **最佳化時間發散**：短期推動力遠超長期時觸發，捕捉最佳擾動敏感性。 |
| 20 | `f_afd_tropopause_pv_intrusion` | 砸盤動能被被動買單吸收 | **對流層頂侵入**：砸盤動能被被動買單吸收的信號，捕捉破底翻效應。 |
| 21 | `f_afd_vortex_dipole_ejection` | AM/PM 張量 trace | **渦旋偶極子彈射**：早盤/尾盤流動張量的正交性，捕捉日內動能轉換。 |
| 22 | `f_afd_vortex_line_slippage` | 早盤砸盤→中盤吸籌→尾盤拉升 | **渦線滑移**：早盤砸盤→中盤吸籌→尾盤拉升的三因子捕捉。 |
| 23 | `f_afd_vortex_stretching_dot` | 流動性梯度與價格變動內積 | **渦度拉伸**：向上價格與流動性真空梯度的內積。 |
| 24 | `f_afd_vorticity_river_tilting_jacobian` | 50-tick bin Jacobian 行列式 | **渦度河流傾斜 Jacobian**：量價流形拉伸的 Jacobian 行列式。 |
| 25 | `f_afd_wave_mean_flow_jacobian` | 5min/30min 時間尺度 Jacobian | **波與平均流 Jacobian**：5分鐘/30分鐘尺度的 Jacobian 行列式，捕捉短波動能反饋。 |
