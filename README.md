# AICSZL2026

AICSZL2026 是一个面向 A 股日频研究的本地工作流。当前版本使用 Tushare 下载原始数据，以 DuckDB 保存 raw、feature 和 target 数据，使用 LightGBM 训练回归模型，输出 prediction/blend pickle，并通过可替换的 Qlib 0.9.7 adapter 运行 Top-K 回测 POC。

## 环境与安装

要求 Python 3.11。建议在 PowerShell 中创建虚拟环境并以 editable 模式安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyqlib==0.9.7` 的依赖较多，首次解析和安装可能需要较长时间。如果当前环境已经安装了 `pyproject.toml` 中的全部运行依赖，可只安装本项目入口：

```powershell
python -m pip install -e . --no-deps
```

安装后验证：

```powershell
aicszl --help
python -c "import qlib; print(qlib.__version__)"
```

Qlib 版本应为 `0.9.7`。

## 配置与 Tushare token

默认配置位于 `configs/settings.yaml`：

```yaml
project:
  start_date: 20200101
paths:
  raw_db: data/raw.duckdb
  feature_db: data/features.duckdb
  artifacts_dir: artifacts
tushare:
  token_file: token.txt
```

将 Tushare token 单独写入项目根目录的 `token.txt`。该文件已被 Git 忽略，不要提交 token、DuckDB 数据库或生成的 artifacts。

## 下载原始数据

首次运行完整研究链路需要下载六张 v0 raw 表：

```powershell
aicszl raw update `
  --to 20211231 `
  --tables trade_cal,daily,adj_factor,stk_limit,moneyflow,daily_basic `
  --batch-days 10 `
  --retries 8 `
  --retry-sleep-ms 3000
```

### 增量更新全部 raw 数据

已有 raw 数据库时，可以把六张表增量更新到当天或指定截止日期。请先将下面的 `YYYYMMDD` 替换为实际日期（例如 `20260712`），然后执行：

```powershell
aicszl raw update `
  --to YYYYMMDD `
  --tables trade_cal,daily,adj_factor,stk_limit,moneyflow,daily_basic `
  --batch-days 10 `
  --retries 8 `
  --retry-sleep-ms 3000
```

该命令会根据每张表已保存的独立水位继续下载，因此不会从项目起始日期重复获取已经成功写入的数据。真正更新时不要添加 `--dry-run`；该选项只检查参数，不访问 Tushare，也不会写入数据库。

可以先使用 `--dry-run` 检查参数而不访问 Tushare：

```powershell
aicszl raw update --to 20200110 --tables daily --dry-run
```

raw 更新按表保存独立水位，并按批次中间提交；后续使用更晚的 `--to` 日期即可增量更新。

## 增量更新特征

更新特征前，必须先把相关 raw 表更新到目标日期。使用下面的命令查看当前注册的插件、插件输出和更新水位：

```powershell
aicszl feature list
```

更新全部 active 特征插件时，请将 `YYYYMMDD` 替换为当天或所需截止日期：

```powershell
aicszl feature update --to YYYYMMDD
```

也可以按稳定的插件 ID 只更新指定插件：

```powershell
aicszl feature update `
  --to YYYYMMDD `
  --plugins market.raw_fields.v1,limit.high_stop.v1
```

正式选择单位是插件而不是单个输出特征。一个插件内声明的全部特征会一起计算、写入和推进水位。例如 `market.raw_fields.v1` 会同时更新 `market.close.v1` 和 `market.amount.v1`；未选中的插件不会被更新。

可以先执行 dry-run 查看预计更新范围，不计算特征或写入数据库：

```powershell
aicszl feature update `
  --to YYYYMMDD `
  --plugins market.raw_fields.v1 `
  --dry-run
```

特征更新器根据 `trade_cal` 和插件内各输出的水位生成连续交易日，默认按20个交易日分批提交。新插件首次更新时会从项目起始日期 `20200101` 回填；失败重跑会从最后成功批次继续。目标日期为非交易日时，特征水位停在不晚于目标日期的最后一个交易日。

### 价格成交量实验特征组

`market.price_volume_pack.v1` 是一个独立的五输出实验插件，依赖 `daily` 和 `adj_factor`，统一使用20个交易观察的最长回看窗口：

- `market.ret_20d_rank.v1`：20日复权收益率排名。
- `market.reversal_1d_rank.v1`：负1日复权收益率排名，用于测试短期反转。
- `risk.volatility_20d_rank.v1`：20日日收益标准差排名。
- `liquidity.amount_ratio_5d_rank.v1`：当日成交额相对前5日平均成交额的放大倍数排名。
- `market.close_position_20d_rank.v1`：复权收盘价在20日复权最高、最低区间中的位置排名。

只回填或增量更新该插件时，请替换目标日期后执行：

```powershell
aicszl feature update `
  --to YYYYMMDD `
  --plugins market.price_volume_pack.v1
```

新插件首次运行会从 `20200101` 连续回填；最初20个交易日属于合法预热期。五个输出作为同一插件原子更新，其中一个输出或写入失败时整个批次回滚。

静态特征组定义在 `configs/features.yaml`：

- `base_v1`：原有5个基线特征。
- `price_volume_exp_v1`：上述5个新增实验特征。
- `base_plus_price_volume_v1`：基线5个加新增5个，共10个特征。

加入特征库不代表这些特征已经被证明有效。后续应使用相同训练区间、模型参数、预测区间和回测设置，分别比较三个特征组的独立表现与增量价值。

## 特征、target、训练、预测和 blend

当前端到端入口是 `scripts/smoke_task3_to_6.py`。短 smoke 模式使用逗号分隔的交易日：

```powershell
python scripts/smoke_task3_to_6.py `
  --dates 20200109,20200110,20200113,20200114 `
  --n-estimators 5
```

长区间模式需要同时给出训练和预测区间：

```powershell
python scripts/smoke_task3_to_6.py `
  --train-start 20200101 `
  --train-end 20210101 `
  --predict-start 20210101 `
  --predict-end 20220101 `
  --job-name lgb_rank5_2020_train_v1 `
  --blend-name blend_rank5_2021_predict_v1 `
  --n-estimators 50
```

脚本会依次更新内置特征和 `target.ret_5d_rank_pct.v1`，训练 LightGBM，生成 prediction，并产生单模型 blend。

## 生成回测数据并运行 Qlib POC

Task 7 当前通过 Python API 调用。将下面的 `blend_path` 替换为上一步实际输出：

```python
from pathlib import Path

from aicszl.backtests import (
    BacktestRunSettings,
    QlibBacktestAdapter,
    build_score_dataset,
)
from aicszl.config import load_settings
from aicszl.raw import RawStore

settings = load_settings("configs/settings.yaml")
raw_store = RawStore(settings.paths.raw_db, settings.project.start_date)
try:
    dataset = build_score_dataset(
        raw_store,
        blend_path=Path("artifacts/blends/替换为实际文件.pkl"),
        output_dir=settings.paths.artifacts_dir / "backtests",
    )
    result = QlibBacktestAdapter(raw_store).run(
        dataset,
        BacktestRunSettings(topk=50, n_drop=5, initial_cash=100_000_000),
    )
    print(result.report_path)
    print(result.positions_path)
finally:
    raw_store.close()
```

标准 score dataset 使用 T 日收盘后得到的 score；Qlib POC 最早在 T+1 开盘成交。停牌阻止买卖，开盘价触及数值涨停价时禁止买入，触及数值跌停价时禁止卖出。开盘价与涨跌停价使用绝对容差 `1e-6`、相对容差 `0` 比较。当前 POC 不使用 benchmark。

## 输出位置

- `data/raw.duckdb`：Tushare raw 数据和 raw 更新水位。
- `data/features.duckdb`：feature、target、metadata 和更新水位。
- `artifacts/models/`：LightGBM 模型和 metadata JSON。
- `artifacts/predictions/`：预测 pickle。
- `artifacts/blends/`：blend pickle。
- `artifacts/backtests/`：标准 score dataset、Qlib provider、report 和 positions pickle。

这些目录均为本地产物，不应提交到 Git。

## 自动保存模型与随机 baseline 收益图

`scripts/run_model_random_backtest.py` 会运行模型和固定种子随机 baseline，在保存 report、positions 和 metrics 后自动生成 `<output-dir>/equity_curve.png`，并在命令结束时打印图片的绝对路径。

当前 2020–2021 训练、2022–2024 验证实验可执行：

```powershell
python scripts/run_model_random_backtest.py `
  --blend-path artifacts/experiments/train_2020_2021_validate_2022_2024/blends/blend_rank5_validate_2022_2024_v1__a2334dbf.pkl `
  --output-dir artifacts/experiments/train_2020_2021_validate_2022_2024/backtests `
  --topk 50 `
  --n-drop 5 `
  --initial-cash 1000000 `
  --random-seed 42
```

这条命令完成后，收益图固定保存在：

```text
artifacts/experiments/train_2020_2021_validate_2022_2024/backtests/equity_curve.png
```

## 可复用的特征组对比实验

`scripts/run_experiment.py` 是配置驱动的正式实验入口。它会按顺序完成所需特征更新、训练标签构建、多特征组训练与预测、共同股票池对齐、固定种子随机 baseline、共享 Qlib provider、逐组回测、指标汇总和多收益曲线发布。

先使用 dry-run 核对日期、原始数据水位、特征插件和 Qlib 版本；dry-run 不写特征库或实验产物：

```powershell
python scripts/run_experiment.py `
  --experiment configs/experiments/pv_5_vs_10_executable_5d_202607.yaml `
  --dry-run
```

确认后运行正式实验：

```powershell
python scripts/run_experiment.py `
  --experiment configs/experiments/pv_5_vs_10_executable_5d_202607.yaml
```

正式配置比较固定种子随机 baseline、`base_v1` 原5特征和 `base_plus_price_volume_v1` 完整10特征。两个模型使用相同训练标签、有效交易日期、LightGBM 参数和过滤条件。预测完成后，运行器先取所有模型 `(trade_date, ts_code)` 的每日交集，再在共同股票池内重新计算模型百分位排名；随机 baseline 也只在相同排序后的交集上生成，因此三条曲线的可交易候选集合完全一致。

### 可执行的五日收益口径

修正后的五日实验把信号日 `t` 的日线特征视为收盘后才可用，训练标签使用复权开盘价 `open(t+1)` 到 `open(t+6)` 的五交易日收益，回测分数也统一迁移到下一交易日开盘执行。训练期末会自动剔除标签终点与预测期重叠的日期；`TopK=50, NDrop=10` 用于近似五交易日平均持有。

上面的 dry-run 只解析并检查有效训练、预测与执行日期，不写特征库或实验产物；正式命令会写入新的模型、预测、signal score、execution score、回测报告、指标和收益曲线。

每次普通运行都会创建不可覆盖的独立目录：

```text
artifacts/experiments/<experiment-name>/runs/<timestamp>-<config-hash>/
```

其中包含配置快照、`run_manifest.json`、各模型及预测、共同 score、共享 provider、三组 report/positions、`metrics.json`、`metrics.csv` 和 `equity_curve.png`。完成状态只会在全部产物写入并验证后发布。

长任务中断后，使用失败运行目录显式恢复：

```powershell
python scripts/run_experiment.py `
  --experiment configs/experiments/pv_5_vs_10_executable_5d_202607.yaml `
  --resume artifacts/experiments/pv_5_vs_10_executable_5d_202607/runs/替换为失败运行目录
```

恢复前会重新校验配置哈希、特征代码哈希和已记录产物的 SHA-256；损坏阶段及其下游不会被复用。已经完成的运行是只读的，若要重复实验应创建新运行。

多年期实验建议在隐藏后台进程中执行并重定向日志，避免终端会话持续输出：

```powershell
$logDir = "artifacts/logs/pv_5_vs_10_executable_5d_202607"
New-Item -ItemType Directory -Force $logDir | Out-Null
$process = Start-Process `
  -FilePath python `
  -ArgumentList @(
    "scripts/run_experiment.py",
    "--experiment",
    "configs/experiments/pv_5_vs_10_executable_5d_202607.yaml"
  ) `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "$logDir/stdout.log" `
  -RedirectStandardError "$logDir/stderr.log" `
  -WindowStyle Hidden `
  -PassThru
$process.Id
```

运行器不会隐式访问 Tushare；必须先用 `aicszl raw update` 将依赖 raw 表更新到配置中的 `data.feature_cutoff`。运行清单会记录数据水位和代码身份，但不会复制数 GB 的 DuckDB 文件，因此精确归档复现还需要保留对应的 `data/raw.duckdb` 和 `data/features.duckdb`。

## 测试

运行完整测试：

```powershell
python -m pytest -v
```

Qlib 集成测试是真实执行，不会因为缺少依赖而自动跳过。若 Qlib 未正确安装，测试和 adapter 会明确要求安装 `pyqlib==0.9.7`。

## 当前命令边界

`aicszl raw update`、`aicszl feature list` 和 `aicszl feature update` 已完整接通。`target`、`train`、`predict`、`blend` 和 `backtest` 命令组目前仅保留 CLI 骨架，直接调用会返回“not implemented yet”。在这些命令完成正式编排前，请使用上面的 Task 3-6 smoke 脚本和 Task 7 Python API；文档不会把 placeholder 描述为可用命令。
