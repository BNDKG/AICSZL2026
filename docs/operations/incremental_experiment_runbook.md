# AICSZL2026 增量特征实验运行手册

## 1. 目的与适用范围

本手册是后续执行“新增一组特征，与已有特征组训练、预测和回测比较”任务时的强制运行规则。

目标只有三个：

1. 新内容只计算一次；
2. 语义完全相同的旧模型和旧预测必须复用；
3. 每次比较仍重新构建共同样本并运行公平回测；当前 base5 重建为两组，未来候选特征实验为三组。

开始类似任务前，执行者必须先阅读本文件、本次实验 YAML，以及
`docs/research/retired_feature_directions.md`，避免重复已证伪的探索方向。

不得在普通新增特征任务中重新设计缓存架构。只有现有缓存不能满足明确的新需求，且用户批准扩展范围时，才允许修改缓存协议。

## 2. 核心语义

### 2.1 “增量”的含义

项目中的增量不是“从头再跑但最终覆盖相同文件”，而是：

- raw 表按各自 watermark 只抓取缺失日期；
- 特征按 plugin 原子更新，旧 plugin 代码未变且已覆盖截止日时必须 `up-to-date`；
- 新 plugin 只生成自己的全部 outputs；
- 同一模型语义 contract 已存在时跳过数据集组装和 LightGBM fit；
- 同一预测已存在时直接复用，只有同起点尾部缺失日期才续算；
- common universe、provider 和回测按本次比较组重新生成。

### 2.2 允许忽略的别名

以下名称只用于展示，不得导致模型缓存失效：

- experiment name；
- model label；
- `TrainingJob.name`；
- feature-group alias / `x_group`。

命中缓存后，本次 run-local metadata 和预测展示字段应使用当前别名，同时保留 cache source provenance；全局缓存文件不得被改写。

### 2.3 必须导致失效的变化

以下任一项变化都必须产生新模型 key：

- 有序特征列表；
- target；
- 有效训练起止日期；
- filters；
- 模型实现 ID；
- 模型参数；
- 任一特征代码哈希；
- 训练范围内相关 feature/target 数据指纹。

预测范围、预测区间数据指纹或模型 key 变化时，预测缓存必须重新判定 exact、slice、extend 或 miss。

## 3. 特征插件规则

### 3.1 新实验特征必须独立成组

- 新增特征放入独立 plugin，不修改已验证基线 plugin 的 outputs。
- 一个 plugin 的全部 outputs 一起计划、一起更新、一起记录 watermark。
- 当前 `configs/features.yaml` 只保留 `base_v1`。未来确有候选特征实验时，临时新增候选独立组和 `base_v1` 加候选特征的组合组；实验无效后删除其源码、测试和配置，只保留研究结论。
- 第一次实验默认只加入一个受控特征包，不一次性扩展多个无法归因的特征包。

### 3.2 代码哈希

- plugin 使用公式 `sha256(inspect.getsource(func) or repr(func))` 计算代码哈希。
- 不得通过修改通用哈希包装格式让所有旧 plugin 无意义失效。
- 当前注册接口没有整模块哈希参数。如果计算依赖的 helper、SQL 或常量发生语义变化，必须同时修改 plugin 计算函数源码或升级 plugin/feature 版本，使代码哈希明确变化；不得只修改 helper 后继续沿用旧 watermark。

### 3.3 代码变化后的重算边界

当 persisted code hash 与当前 plugin hash 不同：

- 只重置该 plugin 的全部 outputs；
- 从项目首个交易日重新生成该 plugin；
- 不得连带重算代码未变的其他 plugin；
- 不得通过手工改 `feature_meta.code_hash` 假装数据由新算法生成。

如果发现未修改的旧 plugin 计划从 2020 开始重算，应立即停止任务并检查哈希兼容性，不得继续消耗计算资源。

## 4. 模型缓存 v2

### 4.1 目录

```text
artifacts/cache/models/<model_cache_key>/
  model.pkl
  model.meta.json
  cache.json
```

缓存目录是 ignored local artifact，不进入 Git。

### 4.2 命中要求

模型缓存命中必须同时满足：

- schema version 为 2；
- cache key 与当前 semantic contract 完全一致；
- manifest contract 与当前 contract 完全一致；
- model/meta 文件存在；
- SHA-256 校验一致；
- source 为可信的 `trained`；
- `train_rows` 合法。

有效命中时：

- `cache_hit=true`；
- 不组装 pandas 训练集；
- 不调用 LightGBM fit；
- 模型字节 hard-link 或原子复制到本次 run；
- 本次 manifest 写入 `cache_schema_version=2`、key、source 和 rows。

### 4.3 miss

只有 semantic contract 真正变化、缓存不存在或缓存损坏时才允许 miss。miss 时训练一次并原子发布 v2 缓存。

如果预期相同的旧模型出现 miss，必须先比较：

1. 有效训练日期；
2. 有序 features；
3. target 和 filters；
4. model params；
5. feature code hashes；
6. data fingerprint。

不得把“重新训练结果看起来一样”当作缓存正确。

### 4.4 旧产物

- 缺少原始 data fingerprint 的旧模型不得进入全局缓存。
- schema v1 或 `source=legacy` 条目不得冒充 v2 命中。
- 旧失败 run 可以在原 run 内继续使用已完成模型，但必须标记 `cache_mode=unavailable`，不得污染全局缓存。
- 历史 64 位 key 如果没有 v2 schema marker，同样是不可信旧产物。

## 5. 预测缓存 v2

### 5.1 目录

```text
artifacts/cache/predictions/<model_cache_key>/
  <prediction_cache_key>.pkl
  <prediction_cache_key>.json
```

### 5.2 查找顺序

严格按以下顺序：

1. `exact`：区间与指纹完全一致，直接复用全部行；
2. `slice`：有效缓存覆盖更长区间，切出请求区间并发布；
3. `extend`：有效缓存与请求同起点且是严格前缀，只预测缺失尾部交易日；
4. `miss`：不存在兼容缓存，完整预测一次。

### 5.3 审计要求

prediction stage 必须记录：

- `cache_schema_version`；
- `cache_hit`；
- `cache_mode`；
- `cache_key`；
- `cache_source`；
- `rows`；
- `reused_rows`；
- `generated_rows`；
- `generated_range`。

`exact` 应满足 `generated_rows=0` 且 `reused_rows=rows`。

旧预测没有原始数据指纹时不得被全局采用或扩展。旧 run 本地恢复需要明确记录 `unavailable`。

## 6. 标准实验配置

正式入口：

```powershell
python scripts/run_experiment.py `
  --experiment configs/experiments/<experiment>.yaml
```

先 dry-run：

```powershell
python scripts/run_experiment.py `
  --experiment configs/experiments/<experiment>.yaml `
  --dry-run
```

`--dry-run` 只解析和展示计划，不写数据库、不训练、不预测、不回测。

当前 `base5_wide_rebuild_202607.yaml` 的请求边界是：

- train request：`20200101-20240101`；
- predict request：`20240101-20260701`。

5 日 next-open target 会自动做 purge 和执行日映射，因此当前有效边界是：

- train signals：`20200102-20231221`；
- predict signals：`20240102-20260701`；
- execution：`20240103-20260702`。

不得为了让表面日期等于请求日期而取消标签隔离或使用未来数据。

## 7. 每次新增特征实验的执行顺序

### 7.1 执行前

1. 阅读本手册、退役方向记录和实验 YAML。
2. 检查 `git status --short`，保留用户已有脏改动。
3. 检查是否已有实验进程持有 DuckDB；同一个 feature DB 不并发写。
4. 检查实验比较组：当前 base5 重建只含 base5 模型并自动增加随机 baseline；未来候选实验含当前组和新组合组，并自动增加随机 baseline。
5. 运行 focused tests。
6. 运行 dry-run，确认 requested/effective dates、required plugins、特征数量。

### 7.2 预期缓存行为

新增一个特征包时，正常预期必须是：

| 阶段 | 当前 base5 组 | 未来候选组合组 |
|---|---|---|
| 旧 feature plugins | `up-to-date` | `up-to-date` |
| 新 feature plugin | 不适用 | 首次生成一次 |
| model | `cache_hit=true` | 首次 `miss`，以后 hit |
| prediction | `exact` 或 `extend` | 首次 miss/extend，以后 exact |
| common scores/provider/backtests | 重建 | 重建 |

如果旧组训练或预测无解释地 miss，应暂停并定位，不应继续多轮重跑。

### 7.3 后台运行

多年任务使用隐藏后台进程并重定向 stdout/stderr：

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = 'artifacts/experiments/<name>/logs'
Start-Process -FilePath 'python' `
  -ArgumentList @('scripts/run_experiment.py', '--experiment', '<yaml>') `
  -WorkingDirectory (Get-Location).Path `
  -RedirectStandardOutput "$logDir/run-$stamp.stdout.log" `
  -RedirectStandardError "$logDir/run-$stamp.stderr.log" `
  -WindowStyle Hidden `
  -PassThru
```

规则：

- 记录 PID、stdout、stderr；
- 约每 10 分钟检查一次；
- 每次只读 PID、日志尾部约 20-30 行和必要 stage 字段；
- 不高速轮询；
- 不反复读取完整 manifest、完整日志或大型 pickle；
- 用户询问状态时提供当前 stage、最后完成日期、是否报错即可。

## 8. 中断与异常处理

### 8.1 feature update 中断

- 不删除 feature DB；
- 读取 plugin outputs 的 watermark 和 max date；
- 同代码哈希下从共同 watermark 续算；
- code hash 变化时只重算对应 plugin；
- 不直接修改 DuckDB metadata，除非已明确证明是框架 bug 导致的错误哈希迁移，并记录恢复依据。

### 8.2 数据库锁

- 先确认是否为本任务后台进程；
- 不启动第二个写进程；
- 读取现有日志判断活跃或僵死；
- 只终止由当前任务创建且确认异常的进程。

### 8.3 意外缓存 miss

- 不先删缓存；
- 对比 semantic contract 和 fingerprint；
- 校验 schema、manifest、文件 SHA；
- 找到真实差异后再决定是正确失效还是 bug。

### 8.4 缓存损坏

- validator 将损坏条目视为 miss；
- 仅重建对应 key；
- 不清空整个 `artifacts/cache`。

### 8.5 旧 run resume

```powershell
python scripts/run_experiment.py `
  --experiment configs/experiments/<experiment>.yaml `
  --resume artifacts/experiments/<name>/runs/<failed-run>
```

- resume 前验证 config hash、feature code hashes 和 run-local checksums；
- 可信 v2 stage 可进入缓存链路；
- 缺 marker 的旧 stage 仅本 run 使用并标记 unavailable。

## 9. 公平回测规则

- 当前 base5 重建必须产生随机 baseline 与 base5 两条曲线。
- 未来候选特征实验必须产生随机 baseline、当前 base5 组、新候选组合组三条曲线。
- 使用相同预测信号日期、execution 日期、TopK、n_drop、成本和初始资金；
- 所有模型先取共同 `(trade_date, ts_code)` 交集；
- 固定随机种子，并在稳定排序后生成随机分数；
- provider 和全部比较组的 backtest 每个新 run 都重新执行；
- 不用不同股票池的曲线直接比较。

## 10. 完成验收

正式交付前只需一次最终验证，不重复多轮无目的审查：

1. focused tests 通过；
2. full `pytest` 通过；
3. `git diff --check` 通过；
4. run manifest `status=complete`；
5. required plugins 覆盖到 cutoff；
6. 配置声明的全部模型和预测 cache audit 与预期一致；
7. common rows/dates 大于 0；
8. 全部 report 无 NaN；
9. `metrics.json`、`metrics.csv`、`equity_curve.png` 存在；
10. 最终答复展示曲线和主要指标；候选特征实验还需明确其优于、持平或弱于当前组。

## 11. Token 与执行纪律

后续执行者必须遵守：

- 不为普通新增特征任务重写缓存架构；
- 不默认派发多轮重复代码审查；
- 不把完整 manifest、完整 diff 或完整长日志输出到对话；
- 只读取定位问题所需的字段和日志尾部；
- 长任务约 10 分钟轮询一次；
- 后台计算期间不进行无关扫描；
- 没有真实失败时，不重复跑相同实验来“增加信心”；
- 预期命中未发生时，先诊断一次，再修复一次，再验证一次；
- 完成结果后立即展示，不因非阻塞的架构优化延迟交付。

## 12. Git 与产物边界

- source/config/tests/README 可审查；
- `data/`、`artifacts/`、logs、cache、models、predictions、provider、reports、plots 不进入 Git；
- 不删除用户数据、旧 run 或缓存，除非用户明确指定范围；
- 不 stage/commit/push，除非用户明确要求；
- 规划和本地运行记录按现有项目 Git policy 处理。
