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

可以先使用 `--dry-run` 检查参数而不访问 Tushare：

```powershell
aicszl raw update --to 20200110 --tables daily --dry-run
```

raw 更新按表保存独立水位，并按批次中间提交；后续使用更晚的 `--to` 日期即可增量更新。

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
D:\PythonProject\AICSZL2026\artifacts\experiments\train_2020_2021_validate_2022_2024\backtests\equity_curve.png
```

## 测试

运行完整测试：

```powershell
python -m pytest -v
```

Qlib 集成测试是真实执行，不会因为缺少依赖而自动跳过。若 Qlib 未正确安装，测试和 adapter 会明确要求安装 `pyqlib==0.9.7`。

## 当前命令边界

`aicszl raw update` 已完整接通。`feature`、`target`、`train`、`predict`、`blend` 和 `backtest` 命令组目前仅保留 CLI 骨架，直接调用会返回“not implemented yet”。在这些命令完成正式编排前，请使用上面的 Task 3-6 smoke 脚本和 Task 7 Python API；文档不会把 placeholder 描述为可用命令。
