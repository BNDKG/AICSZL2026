from __future__ import annotations

from pathlib import Path

import pandas as pd
import tushare as ts


class TushareRawClient:
    def __init__(self, token: str):
        if not token.strip():
            raise ValueError("Tushare token is empty")
        self.pro = ts.pro_api(token.strip())

    @classmethod
    def from_token_file(cls, token_file: str | Path) -> "TushareRawClient":
        token = Path(token_file).read_text(encoding="utf-8").strip()
        return cls(token)

    def trade_dates(self, start_date: int, end_date: int) -> list[int]:
        df = self.pro.trade_cal(exchange="", start_date=str(start_date), end_date=str(end_date))
        if df.empty:
            return []
        df = df[df["is_open"] == 1].copy()
        return sorted(df["cal_date"].astype(int).tolist())

    def fetch_table(self, table_name: str, trade_date: int, end_date: int | None = None) -> pd.DataFrame:
        method = getattr(self, f"_fetch_{table_name}", None)
        if method is None:
            raise KeyError(f"Unsupported Tushare raw table: {table_name}")
        if table_name == "trade_cal":
            df = method(int(trade_date), int(end_date or trade_date))
        else:
            df = method(int(trade_date))
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].astype(int)
        if "cal_date" in df.columns:
            df["cal_date"] = df["cal_date"].astype(int)
        if "pretrade_date" in df.columns:
            df["pretrade_date"] = df["pretrade_date"].fillna(0).astype(int)
        return df

    def _fetch_trade_cal(self, start_date: int, end_date: int) -> pd.DataFrame:
        return self.pro.trade_cal(exchange="", start_date=str(start_date), end_date=str(end_date))

    def _fetch_daily(self, trade_date: int) -> pd.DataFrame:
        return self.pro.daily(trade_date=str(trade_date))

    def _fetch_adj_factor(self, trade_date: int) -> pd.DataFrame:
        return self.pro.adj_factor(trade_date=str(trade_date))

    def _fetch_stk_limit(self, trade_date: int) -> pd.DataFrame:
        return self.pro.stk_limit(trade_date=str(trade_date))

    def _fetch_moneyflow(self, trade_date: int) -> pd.DataFrame:
        return self.pro.moneyflow(trade_date=str(trade_date))

    def _fetch_daily_basic(self, trade_date: int) -> pd.DataFrame:
        return self.pro.daily_basic(
            ts_code="",
            trade_date=str(trade_date),
            fields="ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps_ttm,dv_ttm,circ_mv,total_mv",
        )
