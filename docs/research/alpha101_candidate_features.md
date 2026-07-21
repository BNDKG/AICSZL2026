# Alpha101 候选特征研究方案

状态：研究设计，尚未实现  
日期：2026-07-20  
适用基线：`base5_wide_rebuild_202607`

## 1. 目的与结论

本文从 Alpha101 中筛选 10 个可以直接建立在 AICSZL2026 现有日频数据和宽表特征架构上的候选特征。筛选目标不是复刻全部 Alpha101，也不是一次性恢复一个大特征包，而是为后续 `base5 vs base5+1` 单特征实验准备可执行、相互尽量不同的研究假设。

这 10 个候选全部只依赖当前已有的 `raw.daily` 和 `raw.adj_factor`，不需要新增行情、行业或基本面数据源。它们主要补充 base5 尚未覆盖的隔夜价格位置、VWAP 偏离、量价相关性、方向持续性和条件反转结构。

候选不等于已证明有效。Alpha101 原论文研究的是另一市场和另一交易环境，本文的“优先级”仅表示数据可得、时点正确、与 base5 具有一定互补性，并且经济含义相对清楚。是否保留只能由统一口径的逐特征实验决定。

## 2. 依据与筛选边界

主要来源：Zura Kakushadze, [101 Formulaic Alphas](https://arxiv.org/abs/1601.00991)；公式见[论文 PDF 的 Appendix A](https://arxiv.org/pdf/1601.00991)。论文将这些信号描述为以日频 OHLC、成交量、VWAP 和收益率为主的短周期公式化 alpha，并定义了横截面排名、时序排名、延迟、差分、滚动相关和滚动协方差等操作符。

本次筛选遵循以下边界：

- 只选择现有数据能够无未来信息计算的公式。
- 信号在交易日 `T` 收盘后形成，仍按当前合同在 `T+1` 开盘执行。
- 不选择依赖行业中性化的公式，因为当前原始仓没有稳定的历史行业分类表。
- 不优先选择 100～250 日长窗口、复杂小数窗口或多层衰减组合，避免第一轮实验成本和过拟合风险过高。
- 不原样恢复已经退役的简单 1 日反转、20 日收益、20 日波动、5 日成交额放大或区间位置特征；只有在量价交互或条件结构明显不同的情况下才保留相近组成项。
- 10 个候选共同属于一个插件、写入同一张 10 列宽表；后续实验仍逐列执行 `base5 vs base5+1`，不把 10 列同时加入模型。

## 3. AICSZL 统一口径

### 3.1 存储和计算单元

- 共同插件：`alpha101.candidate10.v1`。
- 物理存储：一张 `fv_*` 宽表，每个 `(ts_code, trade_date)` 只保存一次，10 个候选分别占一列。
- 全量生成：OHLCV 与复权因子只读取、合并一次，共享复权价格、VWAP、收益率、相对成交量和排名等中间量，10 列一次计算、一次写入。
- 实验隔离：每次模型配置只从这张宽表选择一个候选列，因此存储打包不会改变 `base5 vs base5+1` 的实验口径。
- 性能约束：先用小日期段测速；若推算全量生成超过 1 小时，停止全量任务，先消除重复扫描、逐股票 Python 循环或低效滚动计算。完整 10 列生成硬上限为 1 小时，全部实现、回填、10 次实验和汇总目标约 3 小时。

### 3.2 输入变量

对股票 `i`、交易日 `t` 定义：

- `O/H/L/C = daily.open/high/low/close × adj_factor`，用于跨日价格计算，避免除权除息形成伪跳变。
- `V = daily.vol`，即成交量；Tushare 日线定义的单位是手。
- `A = daily.amount`，即成交额；Tushare 日线定义的单位是千元。单位定义见 [Tushare A 股日线接口](https://tushare.pro/document/2?doc_id=27)。
- `VWAP = (A × 10 / V) × adj_factor`。因为一手为 100 股、成交额为千元，所以未复权均价为 `A × 1000 / (V × 100)`。
- `R = C / delay(C, 1) - 1`。
- `RelVol20 = V / mean(V, 20)`。Alpha101 的部分公式把 `volume` 与 `adv20` 直接比较或相除；本项目明确采用同量纲的 20 日平均成交量，避免单位歧义。

本地数据检查结果：`daily` 共 7,775,601 行，覆盖 `20200102～20260710`；OHLC、`vol`、`amount` 无空值或非正值，且所有日线行都能匹配 `adj_factor`。

### 3.3 操作符

- `CSRank_t(x)`：同一交易日全部有限值的平均百分位排名。
- `TSRank_d(x)`：单只股票当前值在最近 `d` 个有效交易日中的百分位排名。
- `Delta_d(x) = x_t - x_{t-d}`。
- `Corr_d(x, y)`、`Cov_d(x, y)`：单只股票最近 `d` 个有效交易日的 Pearson 相关系数和样本协方差。
- `Min_d/Max_d/Mean_d/Std_d/Sum_d`：最近 `d` 个有效交易日的滚动统计。

所有候选最终再做一次 `CSRank_t`，输出 `[0, 1]` 内的百分位值。这样可以降低不同股价、板块和极端成交量对树模型的无意义尺度影响。滚动相关、协方差和排名必须满足完整窗口；分母为零、窗口方差为零或输入非有限时，该股票当日不产生特征行，不用 0 填充。

下文的 Alpha 编号表示思想来源，不表示逐字符复刻。凡是为了复权、量纲一致性、价格尺度或当前执行时点而做的改造，都以本文列出的 AICSZL 公式为唯一候选定义；后续实现时不得在代码里再悄悄选择另一个“常见版本”。

## 4. 十个候选特征

### 4.1 Alpha#20：隔夜开盘在前日价格区间中的联合位置

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a020_overnight_range_position.v1`
- 优先级：A，建议第 1 个实验
- 历史加载：1 个此前有效交易日
- 原始依赖：`open/high/low/close`

为减弱绝对股价差异，使用百分比缺口改造：

```text
g_high  = (O_t - H_{t-1}) / C_{t-1}
g_close = (O_t - C_{t-1}) / C_{t-1}
g_low   = (O_t - L_{t-1}) / C_{t-1}
signal  = -CSRank(g_high) × CSRank(g_close) × CSRank(g_low)
feature = CSRank(signal)
```

预期价值：base5 没有开盘价、前日高低区间或隔夜跳空结构。该特征把开盘相对前日高、收、低三个位置联合起来，可能区分普通缺口、突破缺口和区间内开盘；这类信息与 5 日收盘收益排名明显不同。A 股涨跌停和隔夜消息较多，因此值得最先验证。

主要风险：开盘价受一字板和流动性约束影响。特征本身不删除涨停样本，仍交给现有回测交易约束处理。

### 4.2 Alpha#2：成交量变化与日内收益的短期相关

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a002_volume_intraday_corr.v1`
- 优先级：A，建议第 2 个实验
- 历史加载：7 个此前有效交易日
- 原始依赖：`open/close/volume`

```text
x = CSRank(Delta_2(log(V)))
y = CSRank((C - O) / O)
signal  = -Corr_6(x, y)
feature = CSRank(signal)
```

预期价值：它不是单独使用涨跌或放量，而是判断“成交量变化”和“日内方向”在最近一周是否稳定同向。负相关较强时可能表示放量下跌、缩量上涨等量价背离，和当前的当日成交额、5 日收益、净流入排名具有不同结构。

主要风险：连续停牌后的首个交易日会形成较大的成交量变化；横截面排名能缓和极值，但实验时仍需检查新股和复牌样本的分布。

### 4.3 Alpha#55：区间位置与成交量的相关性

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a055_range_volume_corr.v1`
- 优先级：A，建议第 3 个实验
- 历史加载：16 个此前有效交易日
- 原始依赖：`high/low/close/volume`

```text
range_pos = (C - Min_12(L)) / (Max_12(H) - Min_12(L))
signal  = -Corr_6(CSRank(range_pos), CSRank(V))
feature = CSRank(signal)
```

预期价值：该特征判断股票在近 12 日区间中的位置是否被成交量确认。与单纯的“价格位于 20 日区间哪里”不同，它研究的是区间位置和成交量在最近 6 日的关系，可能识别缩量上行、放量滞涨或底部放量。

主要风险：12 日最高价等于最低价时分母为零，必须产生缺失而不是人为填值。

### 4.4 Alpha#5：开盘相对历史 VWAP 与收盘偏离 VWAP 的联合压力

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a005_vwap_open_close_pressure.v1`
- 优先级：A，建议第 4 个实验
- 历史加载：9 个此前有效交易日
- 原始依赖：`open/close/volume/amount`

```text
open_gap = (O - Mean_10(VWAP)) / Mean_10(VWAP)
close_gap = (C - VWAP) / VWAP
signal  = CSRank(open_gap) × (-abs(CSRank(close_gap) - 0.5) × 2)
feature = CSRank(signal)
```

预期价值：第一部分描述今日开盘相对近 10 日平均成交成本的位置，第二部分描述收盘偏离当日平均成交成本的程度。它同时包含隔夜定价和盘中资金承接信息，base5 目前没有 VWAP 类变量。

主要风险：这里对论文结构做了 A 股尺度化改造，并把偏离排名中心化，否则百分位排名取绝对值会失去“接近中位数”的含义。必须在文档化公式下固定实现，不能同时测试多个隐含版本。

### 4.5 Alpha#11：VWAP—收盘偏离极值与成交量变化

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a011_vwap_extreme_volume_change.v1`
- 优先级：A，建议第 5 个实验
- 历史加载：3 个此前有效交易日
- 原始依赖：`close/volume/amount`

```text
gap = (VWAP - C) / VWAP
signal = [CSRank(Max_3(gap)) + CSRank(Min_3(gap))]
         × CSRank(Delta_3(log(V)))
feature = CSRank(signal)
```

预期价值：它寻找最近 3 日内 VWAP 偏离的上下极值，并用成交量变化确认。相比单日收盘强弱，这个结构能区分持续性偏离和偶然尾盘波动，也可能与资金流排名形成互补。

主要风险：两个横截面排名相加后再乘量变排名，分布可能偏斜，因此最终横截面排名是必要的。

### 4.6 Alpha#13：价格排名与成交量排名的短期协方差

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a013_price_volume_covariance.v1`
- 优先级：B，建议第 6 个实验
- 历史加载：4 个此前有效交易日
- 原始依赖：`close/volume`

```text
signal  = -Cov_5(CSRank(C), CSRank(V))
feature = CSRank(signal)
```

预期价值：该特征观察股票在市场中的相对价格位置和相对成交活跃度是否共同变化。使用每日横截面排名后，信号更接近量价结构，而不是绝对股价或规模暴露。

主要风险：5 日窗口较短，协方差噪声可能较大；这正是它排在第 6 位而不是前三位的原因。

### 4.7 Alpha#43：相对成交量与 7 日反转强度的交互

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a043_relative_volume_reversal.v1`
- 优先级：B，建议第 7 个实验
- 历史加载：38 个此前有效交易日
- 原始依赖：`close/volume`

```text
volume_state = TSRank_20(RelVol20)
reversal_state = TSRank_8(-(C / delay(C, 7) - 1))
signal  = volume_state × reversal_state
feature = CSRank(signal)
```

预期价值：它不是重新加入一个裸 7 日反转，而是要求反转强度与股票自身的成交量异常状态共同出现。它有机会把 base5 的 5 日收益信息分成“有量”和“无量”两类，从而提供条件非线性。

主要风险：与 `market.ret_5d_rank.v1` 存在部分信息重叠，只有当联合成交量状态带来明确增益时才值得保留。

### 4.8 Alpha#17：价格位置、短期加速度和成交量状态三重交互

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a017_price_accel_volume_state.v1`
- 优先级：B，建议第 8 个实验
- 历史加载：23 个此前有效交易日
- 原始依赖：`close/volume`

```text
price_state = -CSRank(TSRank_10(C))
acceleration = CSRank(Delta_1(Delta_1(log(C))))
volume_state = CSRank(TSRank_5(RelVol20))
signal  = price_state × acceleration × volume_state
feature = CSRank(signal)
```

预期价值：三个组成项分别表达自身价格位置、极短期方向变化和相对成交量状态。只有三者组合才产生高信号，可能为 LightGBM 提供 base5 中缺少的显式交互结构。

主要风险：三重乘积更容易稀释信号，且与模型本身的非线性交互能力有一定重复，因此优先级低于更直接的 A20、A2 和 A55。

### 4.9 Alpha#30：三日方向持续性与成交量占比

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a030_direction_volume_persistence.v1`
- 优先级：B，建议第 9 个实验
- 历史加载：19 个此前有效交易日
- 原始依赖：`close/volume`

```text
direction = sign(R_t) + sign(R_{t-1}) + sign(R_{t-2})
persistence = 1 - CSRank(direction)
volume_share = Sum_5(V) / Sum_20(V)
signal  = persistence × volume_share
feature = CSRank(signal)
```

预期价值：方向持续性和近期成交量占比共同描述短趋势是否得到交易活跃度支持。它比单纯 5 日成交额放大多了方向状态，且使用成交量占比而不是再造一个裸成交额比。

主要风险：仍与已经退役的成交额放大方向相近，因此必须靠单特征实验给出明显证据，否则直接淘汰。

### 4.10 Alpha#40：高价波动与量价相关性的交互

- 共同插件：`alpha101.candidate10.v1`
- 建议输出：`alpha101.a040_high_vol_volume_corr.v1`
- 优先级：C，建议第 10 个实验
- 历史加载：10 个此前有效交易日
- 原始依赖：`high/volume`

为消除原始价格尺度影响，将高价标准差改为高价对数变化的标准差：

```text
high_vol = Std_10(Delta_1(log(H)))
price_volume_corr = Corr_10(log(H), log(V))
signal  = -CSRank(high_vol) × price_volume_corr
feature = CSRank(signal)
```

预期价值：它衡量高点波动率与成交量是否共同扩张，可能识别放量冲高、分歧加剧和高波动趋势。它不是裸 20 日波动率，而是一个 10 日波动—量价相关交互。

主要风险：仍可能与已退役波动率特征或 Alpha#13 高度相关，所以放在最后验证；若增益只来自提高换手或成本，则不保留。

## 5. 推荐实验顺序

按预期互补性、实现歧义和历史加载成本，建议顺序固定为：

1. Alpha#20：隔夜开盘联合位置
2. Alpha#2：成交量变化—日内收益相关
3. Alpha#55：区间位置—成交量相关
4. Alpha#5：VWAP 开盘与收盘压力
5. Alpha#11：VWAP 极值与量变
6. Alpha#13：价格—成交量协方差
7. Alpha#43：相对成交量条件反转
8. Alpha#17：价格加速度与成交量状态
9. Alpha#30：方向持续性与成交量占比
10. Alpha#40：高价波动与量价相关

这个顺序不是预先认定的收益排名。前几个候选与 base5 的信息差异更大、窗口更短、解释更直接；后几个候选包含更多交互或与退役方向有部分重叠，需要更严格的增量证据。

## 6. 后续实现和验收约束

如果后续批准实现：

1. 新增一个 10 输出插件，以最大 38 个历史观察为边界只加载一次原始面板，禁止逐特征重复查询。
2. 共享中间量只计算一次；优先使用向量化 `groupby/rolling` 或 DuckDB 窗口计算，禁止逐股票 Python 循环。
3. 为 10 个公式分别补充单元测试，并补充窗口边界、缺失/零方差、批次切分等价和整张宽表原子写入测试。
4. 一次性回填 10 列，不改 base5 的五个原始特征；全量生成超过 1 小时即视为性能不合格，必须先优化再继续。
5. 分别创建或生成 `base5_plus_aXXX` 实验配置，每次只加入一列，正式执行相同训练、预测、下一开盘执行和回测合同。
6. 同时比较累计收益、年化收益、Sharpe、最大回撤、换手率和交易成本；不能因单一指标改善就保留。
7. 十次独立实验都以固定 base5 为参照，失败候选不进入其他候选的基线。
8. 最后删除失败列，以胜出列重建最终宽表并执行一次组合验证；完整流程目标约 3 小时。

本文仅完成候选分析和实现设计，没有注册插件、修改 `configs/features.yaml`、生成特征数据或启动任何实验。
