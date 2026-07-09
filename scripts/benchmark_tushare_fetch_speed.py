from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import tushare as ts

from aicszl.raw.tushare_client import TushareRawClient


def main() -> int:
    args = parse_args()
    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("token file is empty")

    pro = ts.pro_api(token)
    dates = open_trade_dates(pro, args.start, args.end)
    if args.limit_dates:
        dates = dates[: args.limit_dates]
    if not dates:
        raise SystemExit("no open trade dates in requested range")

    print(f"table={args.table} dates={dates}")
    print()

    old_fetch = old_api_fetcher(pro, args.table)
    current_client = TushareRawClient(token)

    case_map = {
        "old_style_sleep": (
            f"old_style_sleep_{args.old_sleep_ms}ms",
            lambda date: old_fetch(date),
            args.old_sleep_ms,
        ),
        "old_style_no_sleep": ("old_style_no_sleep", lambda date: old_fetch(date), 0),
        "current_client": (
            "current_client",
            lambda date: current_client.fetch_table(args.table, date),
            0,
        ),
    }
    selected_cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    for index, case in enumerate(selected_cases):
        if case not in case_map:
            raise SystemExit(f"unknown case: {case}")
        if index > 0 and args.pause_between_cases_ms > 0:
            time.sleep(args.pause_between_cases_ms / 1000)
        name, fetch, sleep_ms = case_map[case]
        run_case(name=name, dates=dates, fetch=fetch, sleep_ms=sleep_ms)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old and current Tushare fetch speed.")
    parser.add_argument("--table", default="adj_factor", choices=["daily", "adj_factor", "stk_limit", "moneyflow", "daily_basic"])
    parser.add_argument("--start", type=int, default=20200102)
    parser.add_argument("--end", type=int, default=20200110)
    parser.add_argument("--limit-dates", type=int, default=5)
    parser.add_argument("--old-sleep-ms", type=int, default=400)
    parser.add_argument(
        "--cases",
        default="old_style_sleep,old_style_no_sleep,current_client",
        help="Comma-separated cases: old_style_sleep,old_style_no_sleep,current_client",
    )
    parser.add_argument("--pause-between-cases-ms", type=int, default=1000)
    parser.add_argument("--token-file", default="token.txt")
    return parser.parse_args()


def open_trade_dates(pro, start: int, end: int) -> list[int]:
    df = pro.trade_cal(exchange="", start_date=str(start), end_date=str(end))
    df = df[df["is_open"] == 1].copy()
    return sorted(df["cal_date"].astype(int).tolist())


def old_api_fetcher(pro, table: str) -> Callable[[int], pd.DataFrame]:
    if table == "daily":
        return lambda date: pro.daily(trade_date=str(date))
    if table == "adj_factor":
        return lambda date: pro.adj_factor(trade_date=str(date))
    if table == "stk_limit":
        return lambda date: pro.stk_limit(trade_date=str(date))
    if table == "moneyflow":
        return lambda date: pro.moneyflow(trade_date=str(date))
    if table == "daily_basic":
        return lambda date: pro.daily_basic(
            ts_code="",
            trade_date=str(date),
            fields="ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps_ttm,dv_ttm,circ_mv,total_mv",
        )
    raise ValueError(table)


def run_case(
    name: str,
    dates: list[int],
    fetch: Callable[[int], pd.DataFrame],
    sleep_ms: int,
) -> None:
    per_call_ms: list[int] = []
    total_rows = 0
    errors = 0
    started = time.perf_counter()
    for index, date in enumerate(dates):
        if index > 0 and sleep_ms > 0:
            time.sleep(sleep_ms / 1000)
        call_started = time.perf_counter()
        try:
            df = fetch(date)
            call_ms = int((time.perf_counter() - call_started) * 1000)
            rows = len(df)
            total_rows += rows
            per_call_ms.append(call_ms)
            print(f"{name} {date} rows={rows} fetch_ms={call_ms}")
        except Exception as exc:
            call_ms = int((time.perf_counter() - call_started) * 1000)
            errors += 1
            print(f"{name} {date} ERROR fetch_ms={call_ms} error={type(exc).__name__}: {exc}")

    total_ms = int((time.perf_counter() - started) * 1000)
    avg_ms = int(statistics.mean(per_call_ms)) if per_call_ms else -1
    median_ms = int(statistics.median(per_call_ms)) if per_call_ms else -1
    print(
        f"{name} summary dates={len(dates)} rows={total_rows} "
        f"errors={errors} total_ms={total_ms} avg_fetch_ms={avg_ms} "
        f"median_fetch_ms={median_ms}"
    )
    print()


if __name__ == "__main__":
    raise SystemExit(main())
