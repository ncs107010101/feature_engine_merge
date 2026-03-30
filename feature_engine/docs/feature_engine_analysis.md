# Feature Engine 完整特徵分析報告（含邏輯說明）

> **總計 318 個特徵**，分佈在 9 個主要模組中（Alpha v7 新增 25 個，Alpha v8 新增 3 個，Alpha v9 新增 3 個，Alpha v10 新增 4 個，Alpha v11 新增 3 個，Alpha v12 新增 4 個，Alpha v13 新增 13 個，Alpha v14 新增 5 個，Alpha v15 新增 7 個，Alpha v16 新增 5 個，Alpha v17 新增 26 個，Alpha v18 新增 4 個）

---

## 一、Single Stock Daily（39 個特徵）

**資料集**：台股日頻行情（價量、基本面、籌碼、法人、融資券）

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_sma_20` | `收盤價` 20日 SMA | `收盤價` | rolling mean(20) | **趨勢錨點**：20日均線作為中期趨勢基準。 |
| 2 | `f_ema_12` | `收盤價` 12日 EMA | `收盤價` | ewm(span=12) | **短期趨勢追蹤**：捕捉 2-3 週的短期動能。 |
| 3 | `f_chip_large_holder_pct` | 大戶持股比例 | `超過1000張集保占比` | ffill | **籌碼集中度**：反映主力長期佈局。 |
| 4 | `f_day_trade_pct` | 當沖佔比 | `_dt_pct` | 直接透傳 | **投機情緒**：反映短線熱門程度。 |
| 5 | `f_turnover_rate` | 周轉率 | `成交量/流通股數` | 直接透傳 | **換手活躍度**：衡量市場關注度。 |
| 6 | `f_vol_cv` | 量能變異係數 | `成交量` | rolling std/mean | **量能穩定度**：衡量資金進出節奏。 |
| 7 | `f_short_change_roc` | 融券3日回報率 | `融券餘額` | pct_change(3) | **空方情緒**：反映軋空或看空共識。 |
| 8 | `f_open_gap` | 跳空幅度 | `開盤/前收` | safe_div | **隔夜衝擊**：反映開盤預期修正。 |
| 9 | `f_ret_5d` | 5日報酬率 | `收盤價` | pct_change(5) | **短期動能**：常見的週動能指標。 |
| 10 | `f_intraday_zscore` | 日內報酬 Z | `(收-開)/開` | zscore(100) | **日內異常性**：相對於歷史的日內強度。 |
| 11 | `f_close_loc_smooth` | 收盤位置平滑 | `收、高、低` | rolling mean(5) | **控盤強度**：收盤在日內高低的位階。 |
| 12 | `f_foreign_flow_zscore` | 外資流入 Z | `外資買賣/量` | zscore(20) | **外資邊際觀點**：外資異常買賣偵測。 |
| 13 | `f_fund_pe` | 本益比 | `本益比(TEJ)` | 原始值 | **價值位階**：橫向與縱向估值比較。 |
| 14 | `f_hl_range_ewm` | 低波動排名 | `振幅` | cs_rank(ewm) | **波動壓縮**：捕捉爆發前蓄勢。 |
| 15 | `f_retail_heat_composite` | 散戶過熱度 | 多指標 | 加權 rank | **反向指標**：散戶過熱預示頂部。 |
| 16 | `f_value_composite` | 綜合價值因子 | 估值多欄 | 加權 rank | **多維估值**：均衡價值評分。 |
| 17 | `f_div_antiretail` | 法人散戶分歧 | 法人 vs 散戶 | 加權 rank | **主力建倉**：法人買散戶賣的共振。 |
| 18 | `f_reversal_composite` | 複合反轉因子 | 短期多指標 | 加權 rank | **均值回歸**：過度反應後的修復。 |
| 19 | `f_residual_momentum` | 殘餘動能 | 多期 ret | ATR 標準化 | **超額動能**：剔除趨勢後的強度。 |
| 20 | `f_high_low_range_expansion` | 波幅擴張 | MAD 5/20 | ratio | **趨勢啟動**：波動由縮轉張。 |
| 21 | `f_heat_lag` | 熱度預測力 | 熱度、報酬 | corr(lag1, ret) | **情緒慣性**：熱度對報酬的引導。 |
| 22 | `f_alpha_accel` | 報酬加速度 | ret diff2 | zscore(20) | **趨勢拐點**：動能的二階導數。 |
| 23 | `f_chip_concentration` | 臨界籌碼集中 | HHI zscore | 非線性增強 | **主力完成信號**：籌碼極度集中。 |
| 24 | `f_contraction_tension` | 波動收縮蓄勢 | ATR ratio, vol | ratio × log | **不確定性能量**：收縮中的放量。 |
| 25 | `f_vol_elasticity` | 波動彈性 | ATR 5/20 | 多項式 | **爆發捕捉**：波動度的爆炸成長。 |
| 26 | `f_price_migration` | 價格遷移距離 | Σ|dp| | ATR zscore | **區間突破**：價格位階跳變強度。 |
| 27 | `f_phase_diff` | 相位差 | ret 5/20 | 殘差 zscore | **短期超前**：動能領先趨勢。 |
| 28 | `f_sector_resonance` | 量價共振強度 | ret, vol | ret × vol_ratio | **特徵品質**：量價配合的有感度。 |
| 29 | `f_limit_edge` | 漲跌停邊緣壓力 | 距邊界 | distance × vol | **極端測試**：邊界處的壓力測試。 |
| 30 | `f_close_to_high_slope` | 收盤高點趨勢 | (H-C)/(H-L) | slope(5) | **收盤品質**：收在高點的穩定性。 |
| 31 | `f_price_range_contraction_days` | 窄幅波動天數 | range | count(<0.5x) | **靜默期長度**：長時間盤整蓄信。 |
| 32 | `f_price_volume_corr_20d` | 價量相關性 | ret, vol | rolling corr | **趨勢健康度**：量價是否齊揚。 |
| 33 | `f_short_squeeze_potential` | 軋空潛力 | `券資比` | ratio | **軋空風險**：券額佔比極高時的爆發力。 |
| 34 | `f_vol_accel` | 成交量加速度 | `成交量` | diff2 / zscore | **爆量啟動**：量能變化的二階導數。 |
| 35 | `f_volume_surprise` | 成交量驚奇度 | `成交量` | vol / MA20 | **異常量能**：相對均值的異常放量。 |

---

## 二、Single Stock Tick（110 個特徵）

**資料集**：逐筆成交（DealPrice, DealCount, PrFlag, DealTimeSecond）

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_tick_mom` | 主動買佔比 | `raw_tick_mom` | `−z(40)→ewm(3)` | **反向tick動能**：過熱則反向看空。 |
| 2 | `f_vpin` | 知情交易量 | `raw_vpin` | `z(60)→ewm(5)` | **VPIN**：知情交易者比例，流動性風險指標。 |
| 3 | `f_trade_skew` | 成交回報偏度 | `raw_trade_skew` | `−z(60)` | **成交偏度**：大單介入後的短期回調預期。 |
| 4 | `f_close_pressure` | 尾盤收盤壓力 | `raw_close_pressure` | `−z(15)→ewm(3)` | **尾盤壓力**：捕捉主力尾盤作價後的修復。 |
| 5 | `f_tk_surge_1min_count` | 急拉頻率 | `raw_surge_count` | `z(60)` | **急拉偵測**：吸引跟風後的出貨風險。 |
| 6 | `f_tk_plunge_1min_count` | 急跌頻率 | `raw_plunge_count` | `z(60)→ewm(3)` | **恐慌偵測**：急跌後的反彈機會。 |
| 7 | `f_tk_morning_surge_ret` | 早盤衝高幅度 | `raw_morning_surge` | `z(60)→ewm(5)` | **早盤試探**：搶開後的假突破辨識。 |
| 8 | `f_tk_high_zone_active_sell_ratio` | 高位主動賣出 | `raw_high_zone_sell` | `ewm(5)→ts_rank(120)` | **趁高出貨**：高價區的積極賣壓。 |
| 9 | `f_tk_trade_intensity_max` | 最大成交密度 | `raw_intensity_max` | `z(60)` | **爆量峰值**：演算法交易觸發信號。 |
| 10 | `f_tk_large_vwap_dev` | 大單成本偏離 | `raw_large_vwap_dev` | `z(60)→ewm(5)` | **成本位階**：大單追價或低吸偵測。 |
| 11 | `f_tk_failed_surge_trap` | 假突破陷阱 | `raw_failed_surge` | `−ewm(20)→rank` | **多頭陷阱**：拉高回落後的套牢壓力。 |
| 12 | `f_tk_dip_buying_under_plunge` | 急跌逢低承接 | `raw_dip_buy` | `z(60)→ewm(5)` | **智慧買盤**：急跌時的法人吸籌。 |
| 13 | `f_tk_morning_trap_severity` | 早盤陷阱深度 | `raw_morning_trap` | `ewm(5)→ts_rank(120)` | **誘多嚴重度**：早盤套牢盤的殺傷力。 |
| 14 | `f_tk_active_sell_above_vwap` | VWAP上方出貨 | `raw_sell_above_vwap` | `ewm(5)→ts_rank(120)` | **獲利了結**：在成本線以上的出貨。 |
| 15 | `f_tk_large_sell_ratio` | 大單賣出佔比 | `raw_large_sell` | `ewm(5)→ts_rank(120)` | **大資金逃離**：主要賣壓來源。 |
| 16 | `f_volbar_ofi_skew` | OFI偏度 | `raw_ofi_skew` | `ewm(20)→rank` | **買賣不對稱**：Bar級別的壓力形態。 |
| 17 | `f_volbar_vwap_trend` | 日內重心趨勢 | `raw_vwap_trend` | `ewm(20)→rank` | **重心位移**：買壓主導的推升。 |
| 18 | `f_dollarbar_volatility` | 金額Bar波動率 | `raw_dbar_vol` | `ts_z(20)→ewm(5)` | **資金敏感度**：相同量能下的波幅。 |
| 19 | `f_timebar_am_pm_divergence` | 上下午分歧 | `raw_am_pm_div` | `ewm(20)→rank` | **情緒輪動**：早盤樂觀 vs 尾盤轉強。 |
| 20 | `f_tk_smart_money_late_vwap_ratio` | 尾盤聰明錢 | `raw_smart_late` | `ewm(20)→rank` | **秘密建倉**：尾盤在高位的佈局。 |
| 21 | `f_tk_buy_sell_trade_size_ratio` | 買賣單量比 | `raw_size_ratio` | `ewm(3)→rank` | **主導方規模**：大戶 vs 散戶規模差。 |
| 22 | `f_tkb_variance_ratio` | 方差比檢定 | `raw_tkb_var_ratio` | `ts_z(20)→ewm(5)` | **市場效率**：趨勢市 vs 震盪市。 |
| 23 | `f_tkb_run_length_max` | 最長連漲長度 | `raw_tkb_run_length` | `−z(20)→ewm(3)` | **連動飽和度**：過度延伸後的反轉預期。 |
| 24 | `f_tkb_reversal_frequency` | 翻轉頻率 | `raw_tkb_reversal_freq` | `z(20)→ewm(3)` | **多空拉鋸度**：高翻轉代表震盪。 |
| 25 | `f_vb_large_trade_domination` | 大單主導度 | `raw_vb_large_domination` | `z(120)→ewm(10)` | **機構介入度**：大單對Bar的控制力。 |
| 26 | `f_vb_active_sell_intensity_var` | 賣壓不穩定度 | `raw_vb_sell_intensity_var` | `ts_z(20)→ewm(5)` | **策略變化**：賣壓的自發性或被動性。 |
| 27 | `f_vb_pinning_probability` | 價格釘住概率 | `raw_vb_pinning_prob` | `ts_z(20)→ewm(5)` | **吸籌釘點**：在特定價位的護盤。 |
| 28 | `f_vb_ofi_volatility` | OFI波動度 | `raw_vb_ofi_vol` | `ts_z(20)→ewm(5)` | **戰況激烈度**：買賣壓力交替頻率。 |
| 29 | `f_vb_vwap_twap_deviation` | VWAP vs TWAP | `raw_vb_vwap_twap_dev` | `ewm(3)→rank` | **追漲品質**：成交是否集中在高位。 |
| 30 | `f_vb_close_to_vwap_deviation` | 收盤相對重心 | `raw_vb_close_vwap_dev` | `ewm(3)→rank` | **尾盤強勢**：收盤價的認可度。 |
| 31 | `f_vb_extreme_return_duration_ratio` | 極端bar時長 | `raw_vb_ext_dur_ratio` | `ts_z(20)→ewm(5)` | **流動性炸彈**：突然跳跌後的成交。 |
| 32 | `f_vb_trade_size_dispersion` | 成交離散度 | `raw_vb_trade_dispersion` | `ts_z(20)→ewm(5)` | **委託分佈**：不均勻交易偵測。 |
| 33 | `f_vb_absorption_ratio` | 吸收比率 | `raw_vb_absorption_ratio` | `ts_z(20)→ewm(5)` | **被動承接**：不推動價格的成交量。 |
| 34 | `f_db_trend_rsquared` | 趨勢穩定度 R² | `raw_db_trend_r2` | `z(20)→ewm(5)` | **路徑純碎性**：線性的單邊市。 |
| 35 | `f_db_vpin_proxy` | DollarBar VPIN | `raw_db_vpin` | `ts_z(20)→ewm(5)` | **重金失衡**：大資金級別的方向。 |
| 36 | `f_db_downside_deviation` | 下行偏差 | `raw_db_downside_dev` | `ts_z(20)→ewm(5)` | **崩跌風險**：負報酬的極端度。 |
| 37 | `f_db_peak_velocity_ratio` | 量能尖峰比 | `raw_db_peak_vel_ratio` | `ts_z(20)→ewm(5)` | **瞬間操作**：主力集中操作時段。 |
| 38 | `f_db_high_velocity_volatility` | 高速Bar波動 | `raw_db_high_vel_vol` | `ts_z(20)→ewm(5)` | **急行軍波幅**：突發事件驅動。 |
| 39 | `f_flow_explosion` | 訂單流爆發 | `imb`, `n` | `\|imb\|×√n` | **量價齊揚爆發**：方向性壓力。 |
| 40 | `f_liquidity_gap` | 流動性缺口 | `spread`, `vol` | `spread × vol` | **枯竭預警**：大幅跳動的先兆。 |
| 41 | `f_order_burst` | 爆量頻率 | `burst_freq` | `N-norm` | **脈衝交易**：量能高度不均。 |
| 42 | `f_breakout_velocity` | 區間突破速度 | `H, L, t` | `zscore(20)` | **逃逸速度**：脫離盤整的動能。 |
| 43 | `f_order_density` | VWAP 密度 | `raw_vwap_density` | `zscore(20)` | **均衡 vs 逃逸**：重心處的博弈。 |
| 44 | `f_momentum_continuity` | 動能持續性 | `up_ticks` | `zscore(20)` | **穩步攀升**：連貫性的買盤。 |
| 45 | `f_density_ratio` | 成交密度比 | `vol, ticks` | `v_peak/t_peak` | **大單化趨勢**：機構 vs 散戶主導。 |
| 46 | `f_eth_nonadiabatic_entropy_rate` | 非預期報酬率變率 | `DealPrice` | `diff(log)` | **非平衡熵**：遠離均衡的速度。 |
| 47 | `f_eth_adiabatic_maintenance` | 分佈維持功率 | `DealPrice` | `Σ(pr-vwap)dx` | **維持功率**：支撐位穩定能量。 |
| 48 | `f_eth_limit_cycle_dissipation` | 震盪環耗散能 | `DealPrice` | `∮P dV` | **耗散能**：盤整區間的能量損耗。 |
| 49 | `f_eth_active_translational_dissipation` | 主動位移耗散 | `DealPrice` | `v²/μ` | **活性耗散**：主動位移的代價。 |
| 50 | `f_flu_instanton_escape_likelihood` | 瞬子逃逸概率 | `DealPrice` | `exp(-S)` | **逃逸中繼**：爆炸式突破預測。 |
| 51 | `f_flu_fluctuation_asymmetry` | 漲跌波動非對稱 | `DealPrice` | `skew` | **情緒單邊化**：非對稱演化偵測。 |
| 52 | `f_flu_tur_bound_saturation` | TUR 飽和度 | `DealPrice` | `TUR` | **市場效率**：不確定性極限。 |
| 53 | `f_flu_time_asymmetry` | 時間對稱破缺 | `DealPrice` | `⟨dx³⟩` | **不可逆趨勢**：時間箭頭方向。 |
| 54 | `f_flu_detailed_balance_breaking` | 細節平衡破缺 | `DealPrice` | `J_matrix` | **循環流**：非平衡態動能。 |
| 55 | `f_att_quasipotential_well` | 擬勢能井深度 | `DealPrice` | `-log P` | **穩定深度**：價格位階穩定性。 |
| 56 | `f_att_saddle_point_barrier` | 鞍點能壘高度 | `DealPrice` | `peak(Pot)` | **突破難度**：阻力區的激發能。 |
| 57 | `f_att_attractor_transition_rate` | 吸引子跳變率 | `DealPrice` | `flip count` | **重塑期**：價值中樞切換頻率。 |
| 58 | `f_att_min_entropy_deviation` | 最小熵路徑偏離 | `DealPrice` | `error` | **操縱偵測**：無效路徑偏離。 |
| 59 | `f_att_phase_space_contraction` | 相空間收縮率 | `DealPrice` | `div(v)` | **確定性**：波動收斂與聚集。 |
| 60 | `f_mac_temperature_mobility` | 市場有效溫度 | `DealPrice` | `Var/Mean` | **激發度**：流動性敏感性。 |
| 61 | `f_mac_kramers_moyal_error` | KM 預測誤差 | `DealPrice` | `error` | **記憶效應**：非馬可夫特徵。 |
| 62 | `f_mac_probability_velocity_circulation` | 概率流環量 | `DealPrice` | `∮v·dl` | **旋渦特徵**：區間洗盤偵測。 |
| 63 | `f_dpt_phase_transition_singularity` | 相變奇點指標 | `DealPrice` | `singularity` | **崩潰臨界**：集體踩踏預警。 |
| 64 | `f_dpt_nonreciprocal_interaction` | 非互易交互作用 | `PrFlag` | `A-At` | **心理博弈**：非中心化反饋。 |
| 65 | `f_dpt_arrhenius_escape_rate` | 逃逸速率 | `DealPrice` | `rate` | **跳出速率**：箱體突破速度。 |
| 66 | `f_dpt_surprisal_derivative` | 驚訝度變率 | `DealPrice` | `d(-log P)` | **瞬間修正**：資訊衝擊力度。 |
| 67 | `f_dpt_subexponential_responsiveness` | 次指數響應度 | `DealPrice` | `Nonlinear` | **敏感爆炸**：非線性反應特徵。 |
| 68 | `f_qit_arnold_web_resonance` | 阿諾德網共振 | `DealPrice` | `Mode ratio` | **相空間穩定**：價位吸引子強度。 |
| 69 | `f_qit_bremsstrahlung_tick_radiation` | 制動輻射碎單 | `DealPrice` | `Frag ratio` | **減速碎單**：恐慌中的零星單。 |
| 70 | `f_qit_decoherence_spin_echo` | 自旋迴聲一致性 | `DealPrice` | `Echo count` | **狀態重構**：起跌點修復概率。 |
| 71 | `f_qit_dense_coding_capacity` | 密集編碼資訊量 | `DealPrice` | `Jump ratio` | **跳檔資訊**：跨檔成交密度。 |
| 72 | `f_qit_dfs_isolation_score` | DFS 空間隔離度 | `DealPrice` | `Inner ratio` | **無噪空間**：點差內交易深度。 |
| 73 | `f_qit_flow_embezzlement_proxy` | 訂單流盜用 | `DealPrice` | `Small push` | **微觀操縱**：小單推升效率。 |
| 74 | `f_qit_gie_entanglement_witness` | 重力誘導糾纏 | `DealPrice` | `Corr(V, dev)` | **價量糾纏**：非古典相關見證。 |
| 75 | `f_qit_gravito_magnetic_induction` | 重力磁感應動量 | `DealPrice` | `Induction` | **磁矩效應**：資金流與位移耦合。 |
| 76 | `f_qit_hawking_evaporation_rate` | 霍金蒸發率 | `DealPrice` | `Evap rate` | **邊界滲透**：高點賣壓輻射。 |
| 77 | `f_qit_holevo_capacity_utilization` | 霍萊沃信道利用 | `DealPrice` | `Efficiency` | **資訊效率**：位移與全路徑比。 |
| 78 | `f_qit_incompatible_observables_product` | 觀測積 | `DealPrice` | `Non-comm` | **不確定性**：位置與動量積。 |
| 79 | `f_qit_lyapunov_ballistic_duration` | 彈道傳播持續 | `DealPrice` | `Max Ballistic` | **連貫買盤**：純主動買序列長。 |
| 80 | `f_qit_magic_state_injection_ratio` | 魔術態注入 | `DealPrice` | `掃透 ratio` | **非克隆趨勢**：瞬間掃單強度。 |
| 81 | `f_qit_majorization_deficit` | 馬約化排序赤字 | `DealPrice` | `Deficit` | **分佈偏離**：早尾盤集中度差。 |
| 82 | `f_qit_otoc_scrambling_rate` | OTOC 擾動率 | `DealPrice` | `decay` | **系統擾動**：資訊擴散速度。 |
| 83 | `f_qit_quantum_zeno_freezing` | 芝諾凍結指數 | `DealPrice` | `Freeze ratio` | **觀察釘住**：頻繁測試引發凍結。 |
| 84 | `f_qit_subadditivity_entropy_gap` | 熵次加成性裂口 | `DealPrice` | `Entropy gap` | **資訊冗餘**：價量跳動互資訊。 |
| 85 | `f_qit_teleportation_fidelity` | 狀態傳態保真度 | `DealPrice` | `Cosine sim` | **狀態重組**：早尾盤相似度。 |
| 86 | `f_qit_virasoro_worldsheet_area` | 弦世界面面積 | `DealPrice` | `Area ratio` | **路徑拓樸**：路徑包圍的有效能。 |
| 87 | `f_qit_wigner_negativity_depth` | 維格納負深度 | `DealPrice` | `Negativity` | **非古典深度**：深度量價背離。 |
| 88 | `f_belief_shock` | 信念衝擊 | `DealPrice` | `ewm(5)` | **反向信念**：大單後的背叛率。 |
| 89 | `f_collapse_direction` | 尾盤崩塌方向 | `DealPrice` | `ewm(3)` | **重心崩塌**：誘多位移偵測。 |
| 90 | `f_contextual_momentum_conflict` | 動能維度衝突 | `DealPrice` | `conflict count` | **共識混亂**：多尺度動能背離。 |
| 91 | `f_early_late_volume_ratio` | 早尾盤量能比 | `DealCount` | `log ratio` | **建倉節奏**：早盤消化 vs 尾盤秘密。 |
| 92 | `f_integer_level_density` | 整數價位吸引力 | `DealPrice` | `sum @ int` | **演算法主導**：忽略心理關卡。 |
| 93 | `f_intraday_reversal_depth` | 日內 V 轉深度 | `DealPrice` | `V-depth` | **洗盤拉升**：淨反彈深度。 |
| 94 | `f_intraday_volatility_clustering` | 波動集群性 | `DealPrice` | `AC(vol)` | **臨界不穩**：大波動連續性。 |
| 95 | `f_large_trade_timing` | 大單時間規律性 | `DealTime` | `HHI(time)` | **目的操作**：大單定時集中爆發。 |
| 96 | `f_order_flow_momentum` | 訂單流動能斜率 | `PrFlag` | `slope(30m)` | **推進力**：主動方單向推進。 |
| 97 | `f_price_bimodality` | 價格雙峰震盪 | `DealPrice` | `modality` | **疊加態**：雙重吸引子拉鋸。 |
| 98 | `f_price_return_asymmetry` | 漲跌時長不對稱 | `DealPrice` | `log ratio` | **風險偏向**：恐慌急殺特徵。 |
| 99 | `f_spread_compression` | 流動性壓縮度 | `Bid/Ask` | `median norm` | **暴風雨前寧靜**：極度流動性。 |
| 100 | `f_tail_session_ofi_zscore` | 尾盤訂單流異常 | `PrFlag` | `zscore(20)` | **定向衝擊**：罕見尾盤流。 |
| 101 | `f_tick_entropy_rate` | 交易規律熵 | `PrFlag` | `entropy` | **預測性**：模式極其規則偵測。 |
| 102 | `f_tick_run_imbalance` | 連續推動力失衡 | `DealPrice` | `MaxRun diff` | **連貫優勢**：多方連貫長度。 |
| 103 | `f_trade_quantization` | 量子化偏移 | `DealCount` | `5-day diff` | **演算法變更**：模式突變。 |
| 104 | `f_volume_acceleration` | 量能隧道效應 | `DealCount` | `2nd deriv` | **趨勢強爆**：以加速度突破。 |
| 105 | `f_volume_price_divergence` | 價量分歧度 | `DealPrice` | `correlation` | **背離預警**：動能衰竭信號。 |
| 106 | `f_vwap_gravity_pull` | VWAP 重力拉力 | `DealPrice` | `AC(dev)` | **均值引力**：回歸 vs 逃逸。 |
| 107 | `f_critical_density_ratio` | 臨界交易密度比 | `vol, ticks` | `zscore(20)` | **臨界吞吐力**：爆量點密度。 |
| 108 | `f_liquidity_gap_intensity` | 流動性缺口強度 | `spread` | `zscore(20)` | **跳動動能**：缺口引發跳空。 |
| 109 | `f_price_breakout_velocity` | 價格突破速度 | `high, low` | `zscore(20)` | **衝擊速率**：阻力位突破。 |
| 110 | `f_second_level_momentum` | 秒級動量連續性 | `tick sequence` | `zscore(20)` | **極高頻趨勢**：秒級推升。 |

---

## 三、Single Stock Broker（31 個特徵）

**資料集**：券商分點買賣（BrokerId, Price, BuyQtm, SellQtm）

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_broker_entropy` | 交易集中度 | `raw_entropy` | `−z(60)` | **主導集中**：少數券商主導信號。 |
| 2 | `f_top5_consist` | 主力一致性 | `raw_consistency` | `z(60)` | **建倉穩定**：同一批券商持續買。 |
| 3 | `f_broker_persist` | 方向持續性 | `raw_persist` | `z(60)` | **操作決心**：連日同向買賣。 |
| 4 | `f_activ_surp` | 活躍度驚奇 | `raw_activity` | `z(60)` | **人氣突變**：新參與者湧入。 |
| 5 | `f_flow_diverg` | 買賣券商分歧 | `raw_diverg` | `z(5)` | **共識分歧**：多空對決共識度。 |
| 6 | `f_net_reversal` | 淨買反轉 | `raw_net_rev` | `rank(120)` | **態度轉向**：資金流突變。 |
| 7 | `f_broker_hhi_buy_sell_diff` | 集中度差 | `raw_hhi_diff` | `ewm(3)` | **吸籌偵測**：買方集中度優勢。 |
| 8 | `f_broker_retail_participation` | 散戶參與度 | `raw_retail_part` | `z(60)` | **籌碼分散**：分點數突增信號。 |
| 9 | `f_broker_daytrade_intensity` | 當沖強度 | `raw_dt_int` | `rank(60)` | **投機強度**：日內套利活躍。 |
| 10 | `f_broker_conviction_buyer_ratio` | 買方集中度 | `Top5 Buy` | `mad_z(20)` | **買進信念**：強力買家主導。 |
| 11 | `f_broker_conviction_seller_ratio` | 賣方集中度 | `Top5 Sell` | `mad_z(20)` | **拋售信念**：集中出貨壓力。 |
| 12 | `f_broker_newcomer_ratio` | 新進者力道 | `Top10 Net` | `mad_z(20)` | **新大戶建倉**：跨券商影響力。 |
| 13 | `f_broker_loyal_inflow` | 忠誠資金流入 | `Large Net %` | `mad_z(20)` | **向心力**：堅定買家強度。 |
| 14 | `f_broker_polarization_index` | 極化指數 | `Std(Net)` | `z(60)` | **博弈強度**：多空分歧激烈。 |
| 15 | `f_broker_hhi_concentration` | 券商數量變化 | `N_active` | `z(60)` | **結構突變**：參與者基數變化。 |
| 16 | `f_newcomer_net_momentum` | 小戶成本偏離 | `Low-mid VWAP` | `z(20)` | **反向指標**：小戶追價熱度。 |
| 17 | `f_smart_money_accel` | 主力加速度 | `d²Net` | `z(20)` | **力道增長**：買入加速度。 |
| 18 | `f_energy_integral` | 累計能量積分 | `ΣNet dτ` | `z(20)` | **蓄能程度**：近期佈局深度。 |
| 19 | `f_direction_consistency` | 操作穩定性 | `sign agree` | `MA(20)` | **持續力**：操作方向連貫性。 |
| 20 | `f_distribution_energy` | 倒貨能量密度 | `sell_freq` | `z(20)` | **出貨警示**：高頻出貨模式。 |
| 21 | `f_mac_field_current_divergence` | 場電流發散 | `NetBuy` | `div J` | **籌碼動態**：鬆動與收籌偵測。 |
| 22 | `f_mac_broker_connectivity_breakdown` | 網絡崩潰度 | `Connectivity` | `Loss coeff` | **共識瓦解**：連動邏輯失效。 |
| 23 | `f_broker_concentration_shift` | 集中度遷移 | `ΔHHI` | `z(20)` | **結構變遷**：相變轉折前兆。 |
| 24 | `f_broker_herding_intensity` | 羊群效應 | `\|2p-1\|` | `z(20)` | **集體情緒**：狂熱或恐懼度。 |
| 25 | `f_consensus_fracture` | 共識斷裂度 | `Trend/Rev` | `z(20)` | **博弈臨界**：力量平衡破裂。 |
| 26 | `f_net_buy_persistence_slope` | 淨買慣性斜率 | `slope(Net)` | `z(60)` | **建倉強勢**：線性買盤推進。 |
| 27 | `f_qit_athermal_free_energy` | 非熱力自由能 | `KL div` | - | **分佈紊亂**：隨機性偏離。 |
| 28 | `f_qit_dilution_factor_inverse` | 稀釋因子倒數 | `HHI` | - | **參與集中**：參與態分佈。 |
| 29 | `f_qit_information_causality` | 資訊因果溢出 | `Ratio` | - | **溢出價值**：界限外的價值。 |
| 30 | `f_qit_liquidity_catalysis` | 流動性催化率 | `MarketMaker%` | - | **造市影響**：不佔位的成交。 |
| 31 | `f_qit_topological_volume_defects` | 拓樸量能缺陷 | `Price dev %` | - | **結構缺陷**：均價偏差量能。 |

---

## 四、HF Profile 特徵（32 個特徵）

**資料集**：券商分點 × 逐筆Tick 聯合（跨維度透視）

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_smart_accumulation_blwVWAP` | 大戶低位吸籌 | `Top10% @ <VWAP` | `ewm(3)→rank` | **聰明錢低吸**：隱蔽建倉。 |
| 2 | `f_retail_trapped_top20` | 散戶高位被套 | `Bot80% @ Top20%` | `ewm(3)→rank` | **追高被套**：過熱信號。 |
| 3 | `f_retail_falling_knife_bot20` | 散戶接飛刀 | `Bot80% @ Bot20%` | `ewm(3)→rank` | **抄底風險**：恐慌接盤。 |
| 4 | `f_lateday_conviction` | 大戶尾盤確信 | `Top20% @ Late` | `ewm(3)→rank` | **隔日預期**：尾盤積極性。 |
| 5 | `f_smart_breakout_top20` | 大戶突破買入 | `Top10% @ Top20%` | `ewm(3)→rank` | **確信突破**：參與高位掃單。 |
| 6 | `f_morning_retail_sel_bot_80%` | 散戶早盤拋售 | `Bot80% @ AM` | `ewm(3)→rank` | **情緒過度**：反向信號。 |
| 7 | `f_morning_smart_buy_top_10%` | 大戶早盤搶買 | `Top10% @ AM` | `ewm(3)→rank` | **當日看多**：開盤佈局。 |
| 8 | `f_large_psell_wall_top_5%` | 大戶掛單賣牆 | `Top5% PassiveSell` | `ewm(5)→rank` | **上方阻力**：控盤出貨。 |
| 9 | `f_aggressor_concentration` | 主動方集中度差 | `HHI_b - HHI_s` | `zscore(20)` | **攻守易位**：主導方權力。 |
| 10 | `f_morning_smart_active_buy` | 早盤大戶主動買 | `Top10% AM Active` | 原始值 | **搶先手**：當日主動掃貨。 |
| 11 | `f_morning_retail_panic` | 早盤小戶拋售 | `Bot80% AM Sell` | 原始值 | **早盤恐慌**：拋售強度。 |
| 12 | `f_large_passive_sell_wall` | 大戶被動賣牆 | `Top10% Passive` | 原始值 | **限價壓力**：阻力強度。 |
| 13 | `f_large_broker_active_buy` | 大券商主動進攻 | `Top10% ActiveB` | 原始值 | **最強動能**：積極搶籌。 |
| 14 | `f_large_broker_passive_buying` | 大券商被動承接 | `Top10% PassiveB` | 原始值 | **下方支撐**：限價接貨。 |
| 15 | `f_concentrated_passive_buy` | 被動買集中度 | `HHI(PassiveB)` | 原始值 | **吸籌深度**：集中建倉偵測。 |
| 16 | `f_eth_conservative_work` | 保守功發散度 | `NetBuy/Path` | ratio | **能量效率**：吸籌品質。 |
| 17 | `f_broker_sync_intensity` | 隱蔽協同買入 | `SyncRate` | zscore(20) | **幕後手**：多券商連動。 |
| 18 | `f_flow_deviation` | 量價流向偏離 | 跨券商資金流 | Nonlinear Dev | **偏離偵測**：資金流與報酬的非同步性。 |
| 19 | `f_retail_trap` | 散戶陷阱深度 | 跨券商籌碼 | Trap Index | **套牢空間**：散戶追高後的浮虧分佈。 |
| 20 | `f_cost_stack` | 主力成本堆疊 | 跨券商成本 | Stack Height | **支撐壓力**：主力持倉形成的位階牆。 |
| 21 | `f_it_secrecy_capacity_rate` | 安全容量率 | `I(X;Y_large) - I(X;Y_small)` / H(T) × net_ratio | `raw_dist_big`, `raw_p_large_buy`, `raw_p_small_buy`, `raw_p_active_up`, `raw_big_net_ratio` | 直接計算 | **安全容量**：大戶與散戶對知情交易貢獻差異，標準化後衡量訊息傳遞效率。 |
| 22 | `f_it_holevo_privacy_premium` | Holevo 資訊溢價 | `χ_up - χ_dn` | `raw_dist_big`, `raw_p_active_up` | 直接計算 | **隱藏信念**：上漲/下跌環境下的大戶資訊內容差異，捕捉主力對方向的隱藏信念。 |
| 23 | `f_it_eavesdropper_confusion_polarity` | 竊聽者困惑度 | `D_KL(P_small || P_big) × net_ratio` | `raw_dist_big`, `raw_dist_small`, `raw_big_net_ratio` | 直接計算 | **散戶被誤導**：散戶與大戶行為分佈的 KL 散度，衡量散戶被訊息誤導的程度。 |
| 24 | `f_it_public_private_kl_direction` | 公私訊息背離 | `KL(Public || Private) × direction` | `raw_p_active_up`, `raw_p_public_buy` | 直接計算 | **訊息不對稱**：掛單偏度與成交偏度的 KL 散度，乘以方向性因子。 |
| 25 | `f_it_directed_info_updown_asymmetry` | 定向資訊不對稱 | `DI_up - DI_down`, rolling z-score | `raw_big_net_ratio`, `raw_p_active_up` | 20日滾動窗口 | **方向偏差**：券商流向對漲跌的定向資訊差異，標準化後消除量級差異。 |
| 26 | `f_it_causal_phi_coefficient` | 因果 φ 係數 | `φ(B, Y)` 在 20 日窗口 | `raw_big_net_ratio`, `raw_p_active_up` | 20日滾動窗口 | **因果關聯**：券商方向與成交方向的 phi 相關係數，衡量兩者間的因果關聯強度。 |
| 27 | `f_it_np_logloss_differential` | 非參對數損失差 | `Gain_up - Gain_dn` | `raw_big_net_ratio`, `raw_p_active_up` | 20日滾動窗口 | **非對稱預測**：券商買入對漲的增益與賣出對跌的增益之差，捕捉資訊預測方向。 |
| 28 | `f_it_time_reversed_di_gap` | 時間反演 DI 差 | `(DI_fwd - DI_rev) / (DI_fwd + DI_rev) × ewm_net` | `raw_big_net_ratio`, `raw_p_active_up`, `raw_ewm_net` | 20日滾動窗口 | **不可逆性**：資訊流的時間不可逆程度，乘以動量權重反映趨勢持續性。 |
| 29 | `f_it_causal_kelly_growth` | 因果凱利增量 | `f* - f_base`, 20日窗口 | `raw_big_net_ratio`, `raw_p_active_up` | 直接計算 | **因果凱利**：條件凱利與基準凱利之差，衡量券商對漲跌的因果影響力。 |
| 30 | `f_it_lautum_penalty_polarity` | 拉頓姆懲罰極性 | `-D_KL(P(B|Y_lag)||P(B)) × net_ratio`, 20日窗口 | `raw_big_net_ratio`, `raw_p_active_up` | 直接計算 | **拉頓姆懲罰**：券商分布偏離均勻態的 KL 散度，乘以淨買比反映主力隱蔽行為。 |
| 31 | `f_it_compression_mismatch_redundancy` | 正規化壓縮冗餘 | `NMI × tanh(skew_B × skew_T)` | `raw_dist_big`, `raw_p_active_up`, `raw_rolling_skew_net`, `raw_rolling_skew_tick` | 直接計算 | **壓縮冗餘**：NMI 乘以偏態積，衡量券商與成交方向的資訊壓縮效率。 |
| 32 | `f_it_conditional_entropy_compression_gain` | 條件熵壓縮增益差 | `Gain_up - Gain_dn` | `raw_big_net_ratio`, `raw_p_active_up` | 直接計算 | **熵增益**：上漲/下跌環境下的對數增益差，捕捉券商方向的預測能力。 |

---

## 五、Alpha v6 Physics（25 個特徵）

**資料集**：逐筆Tick、日頻 OHLC、券商分點（跨維度融合）

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_phy_casimir_vacuum_pressure_gradient` | 真空壓力梯度 | `TickRet, VolRatio` | `Σ x/y` | **無量漂移**：衡量無成交量時的價格「真空」推進力。 |
| 2 | `f_phy_vacancy_induced_gapless_mode` | 空位誘導能隙 | `TickRet, VolRatio` | `Σ y*sign*x²` | **空點衝擊**：稀疏成交區間的高階矩衝擊。 |
| 3 | `f_phy_wignersmith_phase_derivative_q` | 相位導數 | `TickRet, VolRatio` | `sum(mask)` | **相位時滯**：捕捉價格慣性與流向的相位差。 |
| 4 | `f_phy_dcdw_collinear_acceleration` | 共線加速度 | `TickRet, Flow` | `0.5*(x*y+|x*y|)` | **單邊推動**：淨流向與回報的非線性共線加速。 |
| 5 | `f_phy_dislocation_glide_climb_bias` | 位錯滑移偏置 | `Buy/Sell VR` | `log(R_dn/R_up)` | **斜率偏置**：買賣雙方壓力失衡引發的位錯。 |
| 6 | `f_phy_neutrino_energy_degradation_regeneration` | 能量再生 | `Inertia, IntraRet` | `prod` | **動能修復**：大幅震盪後重心回歸的再生能量。 |
| 7 | `f_phy_tidal_trapping_breakout` | 潮汐陷阱突破 | `Trap, IntraRet` | `prod` | **糾結突破**：長期黏附 VWAP 後的爆发性逃逸。 |
| 8 | `f_phy_fractional_charge_excitation` | 分數電荷激發 | `ConsecTicks` | `Σ Vol×Ticks` | **堆疊激發**：同一價位長時間停滯後的能量噴發。 |
| 9 | `f_phy_phase_matching_resonance` | 相位匹配共振 | `NetBuyRatio` | `AC(1m) × NBR` | **多頻共振**：分鐘級波動與日級流向的對齊程度。 |
| 10 | `f_phy_ashtekar_torsion_axial_current` | 扭率軸流 | `Spread, Price` | `Torsion * Vol` | **非線性扭轉**：價格相對於買賣價差的螺旋推進。 |
| 11 | `f_phy_chiral_parity_violating_flux` | 手性通量 | `dSpread, sign` | `flux sum` | **不對稱流**：價差擴張與流向的不守恆耦合。 |
| 12 | `f_phy_parity_transformation_invariant` | 宇稱不變量 | `dSpread, Vol` | `invariant` | **阻尼度量**：市場對對稱價格操作的抵禦力。 |
| 13 | `f_phy_meromorphic_zero_dominance` | 亞純零點主導 | `MidRet, dt` | `Σ ret * dt` | **時間效應**：中間價微小變動的累積時間價值。 |
| 14 | `f_phy_scattering_vs_medium_reservoir_flux` | 散射通量 | `QuoteRet, Vol` | `Σ sign*ret*vol` | **微觀散射**：報價跳動與成交的非線性交互。 |
| 15 | `f_phy_optical_depth_breakthrough` | 光學深度突破 | `Depth, TickRet` | `Σ Depth*Ret` | **密度穿透**：突破委託簿高密度區後的加速。 |
| 16 | `f_phy_landau_peierls_quench` | 朗道驟冷 | `RevFreq, sign` | `rolling sum` | **能量耗散**：極端反轉後的市場情緒冷卻。 |
| 17 | `f_phy_bright_dark_mode_covariance` | 亮暗模態協方差 | `Top10, Bot80` | `-corr * Top10` | **極化同步**：主力與散戶負相關下的行動一致性。 |
| 18 | `f_phy_dark_em_repulsive_force` | 暗電磁斥力 | `NetRatio, Ret` | `-(B-T)*|Ret|` | **極化斥力**：多空分佈兩極化後的背離排斥力。 |
| 19 | `f_phy_entanglement_generation_trajectory` | 糾纏生成軌跡 | `Top5, Ret` | `corr * Top5` | **非定域糾纏**：資金流與價格的同步演化軌跡。 |
| 20 | `f_phy_geodesic_jacobi_convergence` | 測地線收斂 | `CostAdv, NBR` | `sign * |x*y|` | **路徑最優**：成本優勢引發的趨勢收斂。 |
| 21 | `f_phy_geometric_torque_scattering` | 幾何扭矩散射 | `NBR, Dist` | `NBR / Dist` | **邊界吸引**：價格逼近箱體邊界時的扭矩引力。 |
| 22 | `f_phy_interstitial_topological_stress` | 間隙拓撲應力 | `Top5 Buy/Sell` | `Σ stress` | **邊界應力**：大戶在極端價位的防禦性支撐壓力。 |
| 23 | `f_phy_multiplex_coherence_alignment` | 多層相干對齊 | `Score, NBR` | `Score * NBR` | **趨勢共振**：多週期趨勢與主力的相位相干。 |
| 24 | `f_phy_supernova_core_collapse_vector` | 超新星核坍縮 | `Net, PriceDev` | `-(B-T) * dev` | **重心失穩**：籌碼結構失衡引發的價格中心塌陷。 |
| 25 | `f_phy_bic_radiationless_momentum` | 非輻射動量 | `Vol, HL_Ret` | `Vol/HL * Ret` | **有效推進**：扣除震盪損耗後的淨推進能量。 |


---

## 六、Alpha v7 新物理特徵（25 個特徵）

**資料集**：逐筆Tick (`single_stock_tick`)、券商分點 (`single_stock_broker`)、交叉 (`cross_broker_tick`)
**Batch**：20260312 整合批次

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_mhd_magnetic_helicity` | A=cumsum(NetFlow), B=tanh(ΔP×10), H=ΣA·B/(N·max\|A\|) | `DealPrice`, `DealCount`, `PrFlag` | Helicity | **訂單流磁螺旋度**：動量累積與價格方向的螺旋纏繞程度。 |
| 2 | `f_mhd_beltrami_alignment` | cosine_sim(B_bin, J_bin_diff) per 50-tick bin | `DealPrice`, `DealCount`, `PrFlag` | Vector cosine | **貝特拉米對齊**：50-tick bin 的價格趨勢與大單加速度的相似度。 |
| 3 | `f_mhd_vortex_asymmetric_contraction` | log(det_cov_up / det_cov_dn) | `DealPrice`, `DealCount`, `PrFlag` | Log det ratio | **渦旋不對稱收縮**：上漲/下跌 tick 協方差行列式對數比。 |
| 4 | `f_pvg_negative_temperature` | corr(\|P-Open\|, NetFlow) × confinement_60d | `DealPrice`, `PrFlag`, `total_vol` | Corr × rolling | **負溫度微結構**：能量相關性×60日受限度，相關反轉=負溫度信號。 |
| 5 | `f_pvg_spin_glass_susceptibility` | Σ autocorr(spin_bin, lag)×mean_spin per 100-tick | `PrFlag`, `DealCount` | AC sum | **自旋玻璃磁化率**：淨流方向自相關之和×平均方向。 |
| 6 | `f_pvg_high_energy_reference_shift` | log(vol_above_E0 / vol_below_E0) | `DealPrice`, `DealCount` | Log vol ratio | **極端能量參考偏移**：大單 VWAP 以上/以下成交量對數比。 |
| 7 | `f_lrt_fd_asymmetry` | (R_up-R_dn)/(R_up+R_dn), R=\|mean\|/std of dp/vol | `DealPrice`, `DealCount`, `PrFlag` | Efficiency ratio | **漲落耗散非對稱**：上漲/下跌 dp_per_vol 響應效率比。 |
| 8 | `f_lrt_kubo_response_polarization` | 大單後5-tick Δp/vol，比較 buy vs sell | `DealPrice`, `DealCount`, `PrFlag` | Post-shock mean | **庫柏響應極化度**：大買/賣單衝擊後5-tick 價格響應比較。 |
| 9 | `f_lrt_conjugate_coskewness` | E[(V-μV)²(P-μP)]/(σV²·σP) per 50-tick bin | `DealPrice`, `DealCount`, `PrFlag` | Co-skewness | **共軛變數偏態**：50-tick bin 量價 Co-Skewness，揭示非線性耦合。 |
| 10 | `f_crack_creep_to_jump` | 上/下吃檔量 30-tick bin 二階差分差 | `DealPrice`, `DealCount`, `PrFlag` | diff(diff(vol_bin)) | **潛變至斷裂**：累積壓力後突跳，正=向上加速。 |
| 11 | `f_crack_time_reversal_action` | (S_fwd-S_rev)/(S_fwd+S_rev)×sign(Close-Open) | `DealPrice`, `DealCount`, `PrFlag` | Action ratio | **時間反演作用量**：正向與反演時間序列差×方向，不對稱=不可逆趨勢。 |
| 12 | `f_crack_liquidity_collapse` | BuyPr/SellPr median per 50-tick bin | `BuyPr`, `SellPr`, `DealCount` | Advance/retreat ratio | **流動性塌縮**：賣方撤退>買方撤退=正值，流動性向上塌縮。 |
| 13 | `f_sindy_drift_polarity` | mean_ret/std_ret per 30-tick bin | `DealPrice` | t-stat (mean/std) | **SINDy 漂移項極性**：30-tick bin 收益率均值 t-統計量，內生趨勢方向。 |
| 14 | `f_sindy_residual_asymmetry` | skew(dp - EWMA_pred(dp, span=20)) per 20-tick bin | `DealPrice` | skew(residual) | **殘差非對稱性**：EWMA 預測殘差偏態，揭示隱蔽驅動力。 |
| 15 | `f_sindy_phase_transition_gain` | KL(G_1st ‖ G_2nd) × sign(μ2-μ1) | `DealPrice` | KL div × sign | **相變資訊增益**：日內前後分佈 KL 散度×方向，量化制度轉換幅度。 |
| 16 | `f_qm_coarse_graining_loss_bias` | (down_info_loss - up_info_loss) / total_disp | `DealPrice`, `DealCount` | Info loss bias | **粗粒化信息損失偏態**：下跌比上漲損失更多 tick 精細資訊=跌勢更隱蔽。 |
| 17 | `f_qm_density_offdiag_relaxation` | 大/小流量bin交叉相關衰減率 × sign(C_lag1) | `DealCount`, `PrFlag` | Cross-corr decay | **非對角退相干**：大單與小單流量交叉相關的衰減速率。 |
| 18 | `f_qm_measurement_incompatibility` | A_z × (1-\|B_z\|), rolling 20d z-score | `raw_qm_large_broker_net_ratio`, `raw_qm_tick_active_buy_ratio` | z-score product | **量子測量不相容**：大戶淨流與市場追價意願的互補性。 |
| 19 | `f_pol_effective_mass_asymmetry` | (M_dn-M_up)/(M_dn+M_up), M=Σvol/Σ\|ΔP\| | `DealPrice`, `DealCount`, `PrFlag` | Mass ratio | **有效質量非對稱**：下跌慣性>上漲=正值，方向性阻力差異。 |
| 20 | `f_pol_spectral_centroid_shift` | FFT(momentum_bin)的頻率重心 × sign, rolling-20d z-score | `DealPrice`, `DealCount`, `PrFlag` | FFT centroid | **頻譜重心偏移**：動量場主導頻率×方向，20日 z-score 標準化。 |
| 21 | `f_pol_phonon_drag_coefficient` | 大單後10筆小單跟隨率 × log(drag) × 方向 | `DealCount`, `PrFlag` | groupby.apply | **聲子拖曳係數**：大單後小單追隨強度，正=主力帶動散戶追漲。 |
| 22 | `f_geo_lyapunov_drift_vector` | EWMA(ddp, span=50)/std(dp) per 10-tick bin | `DealPrice` | EWMA(diff2) | **Lyapunov 漂移向量**：價格加速度 EWMA，流形上的持續漂移方向。 |
| 23 | `f_geo_entropy_directed_work` | ΣF_bin×dx_bin / (Σ\|dx\|×mean\|F\|), F=(BV-SV)/(BV+SV) | `DealPrice`, `DealCount`, `PrFlag` | Work sum | **熵驅動有向功**：訂單流力量×位移，正=流向與位移方向一致。 |
| 24 | `f_net_markov_flow_asymmetry` | (Σvol×\|ΔP\|_up - Σvol×\|ΔP\|_dn) / total | `DealPrice`, `DealCount`, `PrFlag` | Weighted flow diff | **馬可夫流非對稱**：量加權上漲 vs 下跌位移比，正=向上流動佔優。 |
| 25 | `f_net_antiferro_broker_interaction` | Top5買方集中度-Top5賣方集中度 × 買方淨流比例 | `BuyQtm`, `SellQtm`, `NetBuy`, `BrokerId` | Concentration diff | **反鐵磁性券商交互**：主要買/賣券商間反向交互作用強度。 |

---

## 七、Alpha v8 IT Group 3 泊松通道特徵（3 個特徵）

**資料集**：逐筆成交 (`single_stock_tick`)
**Batch**：20260317 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_it_post_event_directional_impulse` | 大單(≥90th)後10筆Tick的方向衝擊 μ_buy - μ_sell | `DealCount`, `PrFlag` | Post-event mean | **大額成交事件後方向衝擊**：衡量大單後市場反應的方向性，正值代表買方事件後上漲傾向更強。 |
| 2 | `f_it_large_trade_iat_asymmetry` | 大單(≥90th) IAT加速度 (後半/前半 - 1) | `DealTimeSecond`, `DealCount` | IAT ratio | **大額成交到達間距不對稱**：比較買賣雙方大單到達速度，正值代表賣方相對更急促。 |
| 3 | `f_it_broker_poisson_intensity_onesidedness` | 15分時窗買/賣方泊松強度 (λ_buy - λ_sell)/(λ_buy + λ_sell + ε) | `DealTimeSecond`, `DealCount`, `PrFlag` | Poisson rate | **泊松強度單邊性**：衡量買賣雙方成交頻率的偏向，範圍[-1, +1]。 |

---

## 八、Alpha v9 IT Group 5 CTW 通用估計器特徵（3 個特徵）

**資料集**：券商分點 + 逐筆成交（交叉 `cross_broker_tick`）
**Batch**：20260317 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_it_ctw_log_likelihood_ratio` | 3階 Markov LLR: log(P(Y=1\|context)/P(Y=0\|context)), context=(B_{t-3},B_{t-2},B_{t-1}) | `raw_big_net_ratio`, `raw_p_active_up` | Markov LLR | **CTW對數似然比**：使用上下文樹加權法計算3階馬爾可夫鏈的對數似然比，捕捉券商方向歷史對價格方向的預測能力。 |
| 2 | `f_it_di_zscore` | DI滾動z分數 × ewm(net_ratio_big) | `raw_big_net_ratio`, `raw_p_active_up`, `raw_ewm_net` | zscore × ewm | **定向資訊z分數**：1階定向資訊的滾動z分數標準化後，乘以淨買比例的指數加權均值，衡量資訊流向的時間加權強度。 |
| 3 | `f_it_markov_memory_depth_diff` | (D_dn* - D_up*) × ewm_net | `raw_big_net_ratio`, `raw_p_active_up`, `raw_ewm_net` | depth diff × ewm | **馬爾可夫記憶深度差**：最優上漲記憶深度與最優下跌記憶深度之差，乘以淨買比例ewm，捕捉多空記憶不對稱性。 |

---

## 九、Alpha v10 IT Group 6 NRDF 抽象空間特徵（4 個特徵）

**資料集**：券商分點 + 逐筆成交（交叉 `cross_broker_tick`）
**Batch**：20260317 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_it_causal_transfer_rate_deviation` | DI序列斜率 × ewm_net | `raw_di_series`, `raw_ewm_net` | slope × ewm | **因果傳輸率偏離指數**：DI（定向資訊）在20日窗口內的線性斜率乘以當日ewm_net，正值代表因果關係逐日增強。 |
| 2 | `f_it_info_weighted_price_displacement` | Σ ΔH × ΔP (30分鐘分段) | `raw_tick_segments` | segment sum | **資訊加權價格位移**：將Tick按30分鐘分成8段，計算熵降與價格變化的乘積之和，捕捉資訊流動與價格推進的同步強度。 |
| 3 | `f_it_short_long_broker_kl` | KL(q_5d ‖ q_60d) × ewm5 | `raw_dist_big_hist_60d`, `raw_ewm_net` | KL × ewm | **短長期 Broker 分佈 KL**：5日與60日券商淨買分佈的KL散度，乘以5日ewm淨買比例，衡量短期行為偏離長期常態的程度。 |
| 4 | `f_it_broker_distribution_hist_deviation` | KL(today ‖ 20d_med) × tanh(z) | `raw_net_ratio_hist_20d`, `raw_big_net_ratio` | KL × tanh | **Broker 分佈歷史偏離極性**：今日券商分佈與20日歷史中位分佈的KL散度，乘以z分數的tanh值，捕捉分佈異常程度與方向。 |

---

## 十、Alpha v11 IT Group 7 回饋通道與守恆律特徵（3 個特徵）

**輸入**：券商分點 + 逐筆成交（交叉 `cross_broker_tick`）
**Batch**：20260317 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_it_feedback_capacity_utilization_skew` | `CapUtil_up - CapUtil_dn` | `raw_up_info_gain`, `raw_dn_info_gain` | 直接計算 | **上行回饋利用率 skew**：上行趨勢的自強化程度與下跌趨勢的自強化程度之差。 |
| 2 | `f_it_mi_conservation_imbalance` | `(I(B;T) - I(B→T) - I(T→B)) / H(T)` | `raw_broker_entropy`, `raw_tick_entropy` | 直接計算 | **MI 守恆失衡**：衡量信息架構複雜性與VWAP變化的交互作用。 |
| 3 | `f_it_broadcast_channel_entropy_gap` | `(H_small - H_large) × net_ratio_big` | `raw_small_broker_entropy`, `raw_large_broker_entropy`, `raw_net_ratio_big` | 直接計算 | **廣播通道接收者熵差**：散戶與主力交易行為的熵差異。 |

---

## 十一、Alpha v12 生態模擬特徵（4 個特徵）

**輸入**：逐筆成交 (`single_stock_tick`)
**Batch**：20260318 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_fat_tailed_seed_dispersal` | 跨越 30bps 以上的主動買單總量 / 總主動買單量 | `raw_fat_tailed_seed_dispersal` | zscore_rolling(42) | **長尾散播基因流**：衡量大額跳價買單的分散程度，反映市場在极端价格波动时的资金扩散行为。 |
| 2 | `f_masting_predator_satiation` | 爆發分鐘內主動買入量 / 全日被動賣出量 | `raw_burst_buy_vol`, `sell_vol` | zscore_rolling(42) | **Masting 掠食者撐死比**：衡量短時間突襲的力道相對於空方一整天進攻力道的強度，正值代表多方突擊強度大於空方防守。 |
| 3 | `f_mixed_levy_gaussian_dispersion` | 高斯比例 × Lévy 比例 | `raw_mixed_levy_gaussian_dispersion` | zscore_rolling(42) | **Lévy-高斯混合擴散疊加態**：連續小額主動買的比例 (高斯) × 大額跨檔買的比例 (Lévy)，衡量市場在微小波動與大幅跳躍兩種狀態間的混合程度。 |
| 4 | `f_predator_mate_limitation` | plunge_sell_hhi / (plunge_buy_depth + ε) | `raw_plunge_sell_hhi`, `raw_plunge_buy_depth` | zscore_rolling(42) | **掠食者配偶限制**：空方大戶在急跌段（1分鐘 return < -0.3%）找不到對手盤（買方），做空效率大降，衡量空方在急跌時的集中度與深度。 |

---

## 十二、Alpha v13 生態模擬特徵（13 個特徵）

**輸入**：券商分點 (`single_stock_broker`)
**Batch**：20260318 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_allee_reproduction_deficit` | (Top10買入 - 20日均值) / Bottom80%賣出，Z-score(42) | `raw_top10_buy`, `raw_bot80_sell` | rolling_mean(20), zscore_rolling(42) | **族群繁衍臨界赤字**：大戶日買入張數的20日均值回歸程度，除以散戶賣出量，衡量主力回歸與散戶拋壓的失衡程度。 |
| 2 | `f_core_source_contribution_ratio` | 核心源區淨買 / 總淨買，滾動標準化 | `raw_top10_buy`, `raw_top10_sell` | zscore_rolling(42) | **核心源區基因貢獻度**：識別主要資金來源（核心源區）對整體淨買賣的貢獻比率，反映核心主力資金的主導程度。 |
| 3 | `f_cumulative_dose_threshold` | 累積淨買超 / 閾值，Z-score標準化 | `raw_top10_net_buy`, `raw_top10_buy` | rolling_sum(20), zscore_rolling(42) | **累積劑量感染閥值**：追蹤連續買入的累積劑量，當超過歷史閾值時觸發傳染效應，捕捉市場「感染」臨界點。 |
| 4 | `f_dynamic_r0_momentum` | R0估計 × 動量因子，時間加權 | `raw_top10_net_buy`, `raw_retail_vwap` | zscore_rolling(42) | **動態R0動能**：即時估計傳染係數R0並結合動量因子，捕捉疫情式傳播動能的加速或減速。 |
| 5 | `f_herd_immunity_barrier` | (1 - 免疫比例) × 暴露率， barrier 函數標準化 | `raw_top10_buy`, `raw_bot80_sell` | zscore_rolling(42) | **群體免疫屏障效應**：模擬群體免疫水平形成的市場阻力，當免疫比例上升時傳染效率下降，識別市場「免疫」臨界。 |
| 6 | `f_inbreeding_depression_risk` | 相似度指標 × 劣勢表現，偏離度標準化 | `raw_top5_jaccard`, `raw_top5_net_buy` | zscore_rolling(42) | **近親繁殖風險**：測量投資者群體的「基因多樣性」，高度相似性導致市場對新資訊的適應能力下降，預警極端的同質化行為。 |
| 7 | `f_intraguild_predation_skew` | 競爭者間買賣偏度，Top5 vs 次級別 | `raw_top5_net_buy`, `raw_mid10_net_sell` | zscore_rolling(42) | **競爭排除與同類相食**：頂級大戶與次級大戶之間的「捕食關係」，當頂級賣出而次級買入時，代表同類相食效應強化。 |
| 8 | `f_post_latency_infectivity_jump` | 潛伏期後傳染增幅 × 延遲效應 | `raw_top10_net_buy`, `raw_top5_ratio` | rolling_mean(5), zscore_rolling(42) | **潛伏期後傳染跳躍**：模擬疫情傳播的潛伏期效應，識別在初始傳染後的爆發性增強，捕捉市場情緒的延遲反應。 |
| 9 | `f_primary_secondary_infection_div` | 原發感染 vs 次發感染率差異，標準化 | `raw_top10_buy`, `raw_rest_participation` | zscore_rolling(42) | **原發/次發感染背離**：原發（首次買入）與次發（跟隨買入）行為的分離程度，反映市場是處於初始爆發還是跟隨擴散階段。 |
| 10 | `f_resistance_gamma_heterogeneity` | Gamma分佈異質性參數，標準化 | `raw_seller_count`, `raw_top10_net_value` | zscore_rolling(42) | **抗性Gamma異質性**：使用Gamma分佈描述投資者抗性（不願買入）的異質性，高異質性代表市場抵抗力的不均衡分布。 |
| 11 | `f_stoichiometric_nutrient_imbalance` | 資本流入 / 籌碼流出 化學劑量比 | `raw_top10_net_value`, `raw_bot80_net_sell_qtm` | zscore_rolling(42) | **資本-籌碼化學計量失衡**：類比化學反應中的劑量比，當資金（資本）與籌碼供應失衡時，市場將出現類似化學反應的劇烈波動。 |
| 12 | `f_tumor_fingering_breakout` | 腫瘤指狀滲透强度 × 突破閾值 | `raw_top1_vwap`, `raw_rest_vwap`, `raw_daily_max_price` | zscore_rolling(42) | **腫瘤指狀刺透力**：模擬腫瘤細胞指狀入侵 healthy tissue的力學模型，捕捉價格向阻力位「刺透」的爆發力。 |
| 13 | `f_tumor_fingering_instability` | 指狀不穩定性指標，臨界標準化 | `raw_daily_vwap`, `raw_price_range_5d`, `raw_vol_jump` | zscore_rolling(42) | **橫向指狀不穩定性**：測量腫瘤指狀結構的橫向擴張不穩定性，反映市場橫向整理後的突破方向選擇不確定性。 |

---

## 十三、Alpha v14 生態模擬特徵（5 個特徵）

**輸入**：券商分點 + 逐筆成交（交叉 `cross_broker_tick`）
**Batch**：20260319 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_acellular_gap_formation` | `cumsum(spread_on_downtick × top10_net_sell_ratio)`, Z-score(42) | `raw_top10_net_sell_ratio`, `raw_mean_spread_downtick` | cumsum, zscore_rolling(42) | **空方倒貨形成的價差擴大**：空方在下跌時加大拋售力道，同時買方退縮於內盤，形成委賣價差擴大。捕捉主力倒貨時的價格微結構特徵。 |
| 2 | `f_inverse_allee_refuge_accumulation` | `Σ[Top10_BuyQtm × P_refuge(Price)] / 總量`, Z-score(42) | `raw_top10_refuge_ratio` | zscore_rolling(42) | **主力低調吸籌（庇護所效應）**：利用5分鐘低量低波動的「庇護所時段」識別主力隱蔽建倉的行為。高值代表主力在低波動期低調吸籌。 |
| 3 | `f_masting_synchronous_burst` | `爆發期買入HHI集中度`, Z-score(42) | `raw_burst_hhi_top10` | zscore_rolling(42) | **爆發期同步行動集中度**：識別日內主動買入最集中的爆發分鐘，計算Top10在此期間買入的HHI集中度。高值代表主力同步突襲。 |
| 4 | `f_predator_driven_extinction_rate` | `Top5_Sell / PassiveBuy → pct_change(5) → zscore(42)` | `raw_top5_sell_vol`, `raw_passive_buy_vol` | pct_change(5), zscore_rolling(42) | **主力拋售相對於被動承接強度**：Top 5 券商賣出量相對於日內被動承接買量（PrFlag=0）的比率變化，反映主力出貨力道。 |
| 5 | `f_spatial_mixing_refuge` | `Top5 quiet_ratio - Rest quiet_ratio`, Z-score(42) | `raw_top5_quiet_ratio`, `raw_rest_quiet_ratio` | zscore_rolling(42) | **主力避開高波動的超額吸籌**：主力在尾盤平穩期低調建倉，相對於散戶的超額吸籌比例。正值代表主力偏好低波動時段。 |

---

## 十四、Alpha v15 生態模擬特徵（7 個特徵）

**輸入**：券商分點 + 日頻 OHLC（交叉 `cross_broker_daily`）
**Batch**：20260319 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_dilution_effect_defense` | `BuyerCount / Top10NetSellQtm (on dump day)`, EWM(10)→Z-score(42) | `raw_is_dump_day`, `raw_buyer_count`, `raw_top10_net_sell_qtm` | ewm(10), zscore_rolling(42) | **生產者防禦稀釋效應**：砸盤日時淨買入券商數除以大戶淨賣出量的比例，衡量散戶/小戶在主力砸盤時的抵抗能力。 |
| 2 | `f_founder_effect_decay` | `(current_inventory / initial_buy) diff`, Z-score(42) | `raw_is_founder_day`, `raw_top5_net_buy` | zscore_rolling(42) | **創始者效應衰退**：Top5買超券商從建倉到當日累積未平倉佔最初建倉量的比例衰減程度。當此值快速下降，代表先鋒資金在撤退。 |
| 3 | `f_habitat_shifting_speed_deficit` | `speed(VWAP) - speed(Top10_VWAP)`, Z-score(42) | `DailyVWAP`, `raw_top10_buy_vwap` | pct_change, zscore_rolling(42) | **棲地退移速度差**：整體VWAP下移速度與Top10大戶買入VWAP下移速度的差，捕捉主力在價格下移時的相對位置。 |
| 4 | `f_holling_type2_satiation` | `diff²(Top5Sell) × is_new_low_5d`, EWM(10)→Z-score(42) | `raw_is_new_low_5d`, `raw_top5_sell` | diff, ewm(10), zscore_rolling(42) | **掠食者飽食邊界**：Top5賣出量的二階導數，只在短期創新低日計算，識別主力在高點出貨後的動能耗竭。 |
| 5 | `f_inverse_allee_exhaustion` | `(small_buyers_cnt / large_sellers_cnt) slope(10)`, Z-score(42) | `raw_is_new_low_20d`, `raw_small_buyers_cnt`, `raw_large_sellers_cnt` | rolling_slope(10), zscore_rolling(42) | **逆向阿利效應枯竭**：股價創20日新低時，逆勢買入的小型券商家數與順勢賣出的大型券商數的比率趨勢，捕捉散戶逆勢承接行為。 |
| 6 | `f_stefan_boundary_sacrifice_ratio` | `Top10_boundary_buy / Top10_total_buy`, Z-score(42) | `raw_top10_boundary_buy`, `raw_top10_total_buy` | zscore_rolling(42) | **史蒂芬邊界獻祭比例**：大戶在創新高價位（>=daily_high*0.99）的買量佔其總買量的比例，衡量主力在高位是否願意進場。 |
| 7 | `f_super_invader_baseline_defense` | `Top10_baseline_buy / Top10_total_buy`, Z-score(42) | `raw_top10_baseline_buy`, `raw_top10_total_buy` | zscore_rolling(42) | **超級入侵者底線防禦**：大戶在最低價區間（<=daily_low*1.01）的買量佔其總買量的比例，衡量主力在低位是否願意護盤。 |

---

## 十五、Alpha v16 生態模擬特徵（5 個特徵）

**輸入**：逐筆成交 + 日頻 OHLC（交叉 `cross_tick_daily`）
**Batch**：20260319 新增

### cross_tick_daily

Combines tick-level microstructure with daily OHLC data, enabling features that capture the interaction between intraday price dynamics and daily trends. All intermediates are computed at the daily level after aggregating tick data. Used by 5 Alpha v16 features.

**共享中間變數**：
- tick 中間變數：`vwap`, `buy_vol`, `sell_vol`, `total_vol`
- daily OHLC：`收盤價`, `開盤價`
- cross 中間變數：`raw_consecutive_up_days`, `raw_consecutive_up_vol`, `raw_mode_share`, `raw_mode_price`, `raw_patch_thickness`, `raw_push_speed`, `raw_cross_count`, `raw_cross_vol`, `raw_cross_both_vol`, `raw_total_active_buy`, `raw_active_sell`

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_cumulative_dose_exhaustion` | (連續上漲天數 × 期間日均量) / 當日主動賣量 | raw_consecutive_up_days, raw_consecutive_up_vol, raw_active_sell | zscore_rolling(42) | **累積劑量枯竭點**：當連續上漲天數越多且均量越大，但當日主動賣量相對小時，代表買盤即將衰竭。 |
| 2 | `f_density_dependent_dispersal_damping` | Z-score(ModeShare) - Z-score(EscapeDist) → 再Z-score(42d) | raw_mode_share, raw_mode_price, 收盤價 | zscore_rolling(42) | **密度依賴性擴散阻尼**：ModeShare 高代表高密度，EscapeDist 小代表走不動。兩者之差捕捉價格在高密度區間的擴散阻力。 |
| 3 | `f_double_threshold_nonlinear_surge` | 同時跨越VWAP與開盤價的主動買量 / 總主動買量 | raw_cross_both_vol, raw_total_active_buy | zscore_rolling(42) | **雙重閾值非線性躍變**：同時跨越VWAP與開盤價兩個閾值的主動買單量佔比，代表價格突破多個心理關卡的力度。 |
| 4 | `f_limit_cycle_exhaustion_trap` | log((VWAP穿越次數 × 總成交量) / 價格位移幅度) | raw_cross_count, total_vol, 收盤價, 開盤價 | log, zscore_rolling(42) | **動力學極限環耗散枯竭**：VWAP 穿越次數多但價格位移小，代表市場在高頻來回震盪但無有效趨勢，是動能耗散的陷阱區。 |
| 5 | `f_shifting_habitat_patch_deficit` | 推升速度 / (Mode±0.5%成交量佔比 + ε) | raw_push_speed, raw_patch_thickness | zscore_rolling(42) | **斑塊存活赤字**：價格重心推升過快但最大成交量區間極度單薄，代表市場在缺乏流動性支撐下快速位移，是暴漲暴跌的前兆。 |

---

## 十六、Alpha v17 遊戲理論特徵（9 個特徵）

**資料集**：逐筆成交 (`single_stock_tick`)
**Batch**：20260322 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_gt_gradual_fleeing_capitulation` | 連續4-bin主動買下降後尾bin賣出爆量 / sell_ma20 | `DealCount`, `PrFlag` | rolling, ewm, zscore_rolling | **模組1-聰明錢逃離**：連續4個bin主動買遞減後出現極端賣出，代表主力放棄支撐，指標上升→極低報酬。 |
| 2 | `f_gt_forward_induction_signal` | (PM大單買/AM大單買) × AM安靜標記，Z-score(20) | `DealCount`, `PrFlag` | rolling mean, shift(1), zscore_rolling | **模組1-前向歸納**：早盤沉默午後發動，(PM/AM大單比)×AM安靜度，指標上升→極高報酬。 |
| 3 | `f_gt_feigned_ignorance_accumulation` | 內盤(PrFlag=0)被動大買 × 小主動賣 bin均值，Z-score(20) | `DealCount`, `PrFlag` | bin aggregation, zscore_rolling | **模組2-佯裝無知吸籌**：內盤被動大單承接掩護小單主動賣，指標上升→極高報酬。 |
| 4 | `f_gt_feigned_ignorance_distribution` | 外盤(PrFlag=1)小主動買 × 大被動賣 bin均值，Z-score(20) | `DealCount`, `PrFlag` | bin aggregation, zscore_rolling | **模組2-佯裝無知派發**：外盤小單推升掩護大單被動出貨，指標上升→極低報酬（典型拉高出貨）。 |
| 5 | `f_gt_persuasion_suspense_slide` | 沉默streak × 沉默後大單買量，Z-score(20) | `DealCount`, `PrFlag` | rolling mean, shift(1), zscore_rolling | **模組2-說服與懸疑滑落**：長沉默後突襲大單買，指標上升→極高報酬。 |
| 6 | `f_gt_cheap_talk_distraction_trap` | 大單賣量 / 1口買單數，Z-score(20) | `DealCount`, `PrFlag` | bin aggregation, zscore_rolling | **模組3-廉價話語分心陷阱**：1口小單買掩護大單賣，指標上升→極低報酬。 |
| 7 | `f_gt_invidious_distinction_cost` | 大單買VWAP相對bin VWAP溢價 × 大單量，Z-score(20) | `DealCount`, `DealPrice`, `PrFlag` | VWAP calc, zscore_rolling | **模組4-嫉妒區分成本**：大戶願意高於市場均價買入展示資訊優勢，指標上升→極高報酬。 |
| 8 | `f_gt_pecuniary_emulation_trap` | 小單順從昨日大單方向 × 大單反轉 × \|大單淨流\|，Z-score(20) | `DealCount`, `PrFlag` | shift(1), zscore_rolling | **模組4-金錢模仿陷阱**：小單模仿昨日大單方向但大單已反轉，指標上升→極低報酬。 |
| 9 | `f_gt_uninformed_herding_bias` | 小單順從上筆價格方向的比率，Z-score(20) | `DealCount`, `DealPrice`, `PrFlag` | bin aggregation, zscore_rolling | **模組4-未受教導羊群偏誤**：小單盲目跟隨上筆價格方向，指標上升→極低報酬。 |

---

## 十七、Alpha v17 遊戲理論特徵（日頻，6 個特徵）

**資料集**：日頻 OHLCV + 籌碼（`single_stock_daily`）
**Batch**：20260322 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_gt_level_k_retail_harvesting` | `corr(當沖%, ret5) × (-sign(inst_net)) × |inst_net|` → Z-score(20) | `當沖買賣占比`, `收盤價`, `外資/投信/自營淨買` | rolling_corr, zscore_rolling | **Level-k 散戶收割**：當沖熱度與報酬正相關但法人反向操作。法人借散戶慣性收割。 |
| 2 | `f_gt_unraveling_skepticism_failure` | `is_bad(rev_yoy≤0 & ret5≤0) × retail_growth_5` → EWM(3) | `營收`, `收盤價`, `未滿400張集保人數` | pct_change, ewm | **解讀失敗懷疑**：散戶在營收衰退且價格下跌時仍增加持股。代表散戶過度自信。 |
| 3 | `f_gt_delayed_disclosure_moral_hazard` | `1/vol_20d × short_change_5d` → Z-score(60) | `收盤價`, `融券餘額`, `借券賣出餘額` | rolling_std, zscore_rolling | **延遲揭露道德危機**：低波動假象掩護融券餘額上升。代表內部人借券布局。 |
| 4 | `f_gt_reputation_cost_signaling` | `down_3d × inst_change × is_net_positive` → Z-score(20) | `收盤價`, `外資/投信/自營淨買` | rolling, diff, zscore_rolling | **名譽成本信號**：法人在連續下跌後強力買入。顯示其對基本面更有信心。 |
| 5 | `f_gt_strategic_ignorance_demand` | `is_below_ma × is_falling × is_margin_up × margin_change_3` → Z-score(60) | `收盤價`, `SMA20`, `ret5`, `融資餘額` | rolling_mean, pct_change, zscore_rolling | **戰略性無知需求**：散戶在跌破均線且下跌5%時仍增加融資。代表拒絕承認錯誤。 |
| 6 | `f_gt_moral_hazard_leverage` | `(dt_pct + margin_usage) / large_holder_pct` → Z-score(60) | `當沖買賣占比`, `融資餘額`, `流通股數`, `超過1000張集保占比` | zscore_rolling | **道德危機槓桿**：散戶當沖+融資同步放大但大戶退出。槓桿錯配最危險情境。 |

---

## 十八、Alpha v18 行為經濟學特徵（日頻，4 個特徵）

**資料集**：日頻 OHLCV + 籌碼（`single_stock_daily`）
**Batch**：20260328 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_be_asymmetric_confidence_spillover` | `DT_Trend × Bad_News` → Z-score(20) | `_dt_pct`, `_dt_mean_20`, `_ret_3d` | zscore_rolling | **不對稱自信溢出**：當沖比例相對於20日均值的趨勢 × 3日報酬<-3%壞消息。捕捉散戶過度自信，在市場下跌時仍積極參與。 |
| 2 | `f_gt_outgroup_exploitation_bias` | `Outgroup_Influx × Inst_Exploitation` → EWM(5) → Z-score(42) | `_retail_count`, `_inst_net` | ewm_then_zscore | **外團體剝削偏誤**：散戶人數增長率 × 法人淨賣出。捕捉散戶湧入時法人倒貨坑殺外團體的行為。 |
| 3 | `f_be_ego_driven_misreporting` | `trend_down × margin_increase × margin_diff / margin_bal` → Z-score(20) | `收盤價`, `_sma_20`, `_margin_bal` | zscore_rolling | **自我驅動錯誤報告**：股價跌破SMA20且5日報酬<-3%時，融資餘額增加。捕捉散戶因自我 ego 而持續加碼攤平的行為。 |
| 4 | `f_be_frustration_driven_turnover` | `high_expectation × disappointment × turnover_rate` → Z-score(20) | `最高價`, `收盤價`, `開盤價`, `成交量(千股)`, `流通在外股數(千股)` | zscore_rolling | **挫折驅動換手**：股價突破20日新高但收盤翻黑 × 換手率。捕捉投資人挫折情緒引發的爆量換手。 |

---

## 十八、Alpha v17 遊戲理論特徵（券商分點，3 個特徵）

**資料集**：券商分點 (`single_stock_broker`)
**Batch**：20260322 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_gt_payoff_dominant_focal_point` | 聚焦價格（10的倍數）的不同券商數量 × 買入量 → EWM(5) → Z-score(20) | `raw_focal_distinct_brokers`, `raw_focal_buy_vol` | ewm(5), zscore_rolling(20) | **聚焦點 payoff dominant**：多個券商聚焦於整數價位形成支撐。當聚焦點共識越強，代表市場對該價位有強烈共識，後續價格可能向該方向突破。 |
| 2 | `f_gt_expert_vs_popular_divergence` | Top-5 成交量券商淨買 > 0 AND Mid-10 券商淨賣 > 0 → 信號 × 差距 → EWM(5) → Z-score(20) | `raw_top5_net_buy`, `raw_mid10_net_sell` | ewm(5), zscore_rolling(20) | **專家 vs 大眾背離**：當高成交量券商（top-5 by buy volume）淨買但中階券商（rank 6-15）淨賣時，代表大戶正在派發，後續價格可能下跌。（原始版本需 Rolling per-broker metrics，本版本使用日級別近似） |
| 3 | `f_gt_game_of_chicken_standoff` | Buy HHI 和 Sell HHI 都處於60日90百分位且差異很小 → 1/差異 × EWM(3) | `HHI_Buy`, `HHI_Sell` | rolling_quantile(0.9, 60), ewm(3) | **膽小鬼僵局**：買賣雙方都高度集中但勢均力敵。當雙方都認為自己會贏但都不願先讓步時市場將出現高波動僵局。 |

---

## 十九、Alpha v17 遊戲理論特徵（交叉券商日頻，8 個特徵）

**資料集**：券商分點 + 日頻 OHLC（交叉 `cross_broker_daily`）
**Batch**：20260322 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_gt_costly_waiting_attrition` | `rolling_sum(Top5_NetBuy, 5) × is_falling(跌5日報酬<0)` → Z-score(20d) | `raw_top5_net_buy`, `收盤價` | rolling(5).sum, pct_change(5), zscore_rolling(20) | **消耗戰鎖碼**：股價連跌時前5大主力仍持續淨買，滾動5日加總×是否下跌作為 raw，zscore(20)標準化。遊戲理論中展示承受虧損的決心。方向：正向 → 大戶在跌勢中仍堅持 → 後續可能反轉。 |
| 2 | `f_gt_echo_chamber_assortative` | `vol_surge × (1 - top10_ratio)` → EWM(5) → Z-score(20d) | `raw_top10_total_buy`, `成交量(千股)` | rolling(20).mean, ewm(5), zscore_rolling(20) | **同溫層泡泡**：成交量放大但前10大主力參與率下降，代表散戶對敲泡沫。ewm(5)→zscore(20)標準化。方向：正向 → 散戶對敲越活絡 → 極端可能反轉。 |
| 3 | `f_gt_second_order_belief_fading` | `is_breakout(收盤>20日高) × Top5_NetSell` → Z-score(60d) | `raw_top5_net_sell_qtm`, `收盤價`, `最高價` | rolling(20).max, zscore_rolling(60) | **二階信念褪散**：股價突破20日新高時，前5大主力趁機淨賣出收割。is_breakout×Top5_NetSell作 raw，zscore(60)標準化。方向：正向 → 假突破陷阱 → 極端低報酬。 |
| 4 | `f_gt_commitment_device_lock` | `is_strong_up(5日報酬>5%) × is_minimal_sell(賣出量<=20日5%分位)` → EWM(3) | `收盤價`, `_broker_day` (券商日資料) | rolling(20).quantile(0.05), ewm(3) | **承諾機制鎖碼**：近期大漲(5日>5%)且過去20日累積買超最大之前10大券商今日賣出量為0或極低。ewm(3)標準化。方向：正向 → 絕對鎖碼決心 → 極端高報酬。 |
| 31 | `f_gt_gradual_trust_escalation` | `rolling_zscore(trust_test × escalation × price_ok, 20)` | `top5_20d_netbuy`, `top5_20d_positive_days`, `price_stability` | rolling z-score | **消耗戰鎖碼**：Top5 brokers by 20-day cum_net continue buying in falling market. Measures institutional conviction through sustained buying pressure despite adverse price movement. |
| 32 | `f_be_overconfidence_spillover` | `rolling_zscore(condition × top5_5d_netbuy, 20)` | `top5_5d_netbuy`, `past_success`, `today_shock` | rolling z-score | **過度自信溢價**：Top5 brokers from past 5 days with past_success + today_shock. Captures behavioral bias where successful traders overconfidently increase positions after market shocks. |
| 33 | `f_gt_pooling_equilibrium_camouflage` | `ewm_then_zscore(vol_surge × low_concentration, 5, 20)` | `raw_hhi_buy`, `vol_surge` | ewm then z-score | **Pool均衡偽裝**：Volume surge + low concentration. When market is volatile with low concentration, institutional players camouflage trading activity to avoid detection. |
| 34 | `f_gt_incumbent_signaling_exemption` | `ewm_then_zscore(1 / (abs_ret / passive_proxy + ε), 5, 20)` | `top1_10d_netbuy`, `top1_10d_incumbent_buy`, `收盤價`, `開盤價` | ewm then z-score | **現任者豁免**：Top1 incumbent from 10-day history has passive buying. Dominant broker's passive accumulation signals hidden conviction without price impact. |

**共享中間變數** (`preprocess_cross_broker_daily`):
- `raw_top5_net_buy`: Top 5 券商淨買量（使用 nlargest(5, NetBuy) 選擇）
- `raw_top5_net_sell_qtm`: Top 5 券商淨賣量（NetSell = SellQtm - BuyQtm）
- `raw_top10_total_buy`: Top 10 券商總買量

---

## 二十、Alpha v17 遊戲理論特徵（交叉 Tick 日頻，1 個特徵）

**資料集**：逐筆成交 + 券商分點日頻聚合（交叉 `cross_broker_tick`）
**Batch**：20260322 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_gt_information_design_obfuscation` | 50-tick Bin: PrFlag翻轉次數 × DealCount變異係數 → 日頻化為混淆指數 × Top5_NetSell → EWM(10) → Z-score(20) | `raw_obfuscation_index`, `Top5_NetSell` | ewm(10), zscore_rolling(20) | **資訊設計混淆策略**：日內內外盤頻繁翻轉、量忽大忽小（噪音）掩護前五大券商穩定淨賣出。方向：正值 → 主力掩護出貨 → 極端低報酬。 |

**共享中間變數** (`preprocess_cross_broker_tick`):
- `raw_obfuscation_index`: 50-tick bin 的 PrFlag 翻轉次數與 DealCount 變異係數之積的日均值

---

## 二十一、Alpha v17 遊戲理論特徵（交叉逐筆日頻，1 個特徵）

**資料集**：逐筆成交 + 日頻 OHLC（交叉 `cross_tick_daily`）
**Batch**：20260323 新增

| # | 特徵名稱 | 計算方式 | 使用欄位 | 算子 | 邏輯說明 |
|---|---------|---------|---------|------|---------|
| 1 | `f_gt_immediate_disclosure_panic` | 開盤價 < 前日收盤價×0.98 → gap_down × 前500筆賣出量佔比 → zscore_rolling(20) | `開盤價`, `收盤價`, `DealCount`, `PrFlag`, `TotalQty` | zscore_rolling(20) | **模組5-立即揭露恐慌**：跳空低開≥2%且前500筆賣出積極，代表知情交易者立即傾倒持股。指標上升→極端負報酬。 |

**共享中間變數** (`preprocess_cross_tick_daily`):
- 逐筆成交：`DealCount`, `PrFlag`, `TotalQty`（用於排序）
- 日頻 OHLC：`開盤價`, `收盤價`（用於 gap-down 計算）

---

## 算子使用統計

| 算子類型 | 使用次數 | 設計邏輯 |
|---------|---------|---------|
| `zscore_rolling` | ~77 | **標準化**：消除量級差異（v17 新增 22 個）。 |
| `ewm_smooth` | ~38 | **降噪**：保留近期權重。 |
| `ts_rank` | ~25 | **強健化**：對分佈不敏感。 |
| `XS rank` | ~12 | **相對價值**：全市場選拔。 |
| `mad_zscore` | 8 | **極端值處理**：中位數穩健化。 |
| `HHI / Entropy` | ~10 | **結構分析**：集中度量化。 |
| `Physics Operators` | ~35 | **動態建模**：熱力學、場論、量子學算子（v7 新增 10 個）。 |
| `Vectorized bin ops` | ~30 | **向量化**：n-tick bin 操作（Alpha v7 主要方法）。 |
| `Poisson Channel` | 3 | **泊松通道**：大單事件後方向衝擊、IAT不對稱、泊松強度（v8 新增 3 個）。 |
| `NRDF Operators` | 4 | **抽象空間運算**：因果傳輸率、資訊加權位移、短長期KL、歷史分佈偏離（v10 新增 4 個）。 |
| `生態模擬特徵` | 34 | **生態模擬**：長尾散播、Masting 掠食者、Lévy-高斯混合、掠食者配偶限制、Allee 臨界、核心源區、累積劑量、R0 動能、免疫屏障、近親繁殖、同類相食、潛伏期跳躍、原發/次發背離、Gamma 異質性、化學劑量失衡、指狀刺透、指狀不穩定、空方倒貨價差、低調吸籌庇護所、爆發集中度、主力拋售強度、空間混合庇護、生產者防禦稀釋、創始者效應衰退、棲地退移速度差、掠食者飽食邊界、逆向阿利效應枯竭、邊界獻祭比例、底線防禦、累積劑量枯竭點、密度依賴性擴散阻尼、雙重閾值非線性躍變、動力學極限環耗散枯竭、斑塊存活赤字（v12+v13+v14+v15+v16 共 34 個）。 |
| `遊戲理論特徵` | 22 | **遊戲理論**：逐步逃離、前向歸納、佯裝無知吸籌、佯裝無知派發、說服懸疑滑落、廉價話語分心陷阱、嫉妒區分成本、金錢模仿陷阱、未受教導羊群偏誤、Level-k散戶收割、解讀失敗懷疑、延遲揭露道德危機、名譽成本信號、戰略性無知需求、道德危機槓桿、聚焦點 payoff dominant、專家vs大眾背離、膽小鬼僵局、消耗戰鎖碼、同溫層泡泡、二階信念褪散、承諾機制鎖碼（v17 新增 22 個）。 |
