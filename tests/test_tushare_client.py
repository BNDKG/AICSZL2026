import pandas as pd

from aicszl.raw.tushare_client import TushareRawClient


class FakePro:
    def trade_cal(self, **kwargs):
        return pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20200101", "is_open": 0, "pretrade_date": "20191231"},
                {"exchange": "SSE", "cal_date": "20200102", "is_open": 1, "pretrade_date": "20191231"},
            ]
        )

    def daily(self, **kwargs):
        return pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20200102"}])

    def adj_factor(self, **kwargs):
        return pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20200102", "adj_factor": 1.0}])

    def stk_limit(self, **kwargs):
        return pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20200102"}])

    def moneyflow(self, **kwargs):
        return pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20200102"}])

    def daily_basic(self, **kwargs):
        return pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20200102"}])


def make_client():
    client = object.__new__(TushareRawClient)
    client.pro = FakePro()
    return client


def test_trade_dates_returns_only_open_calendar_dates():
    client = make_client()

    assert client.trade_dates(20200101, 20200102) == [20200102]


def test_fetch_trade_cal_normalizes_calendar_dates():
    client = make_client()

    df = client.fetch_table("trade_cal", 20200101, 20200102)

    assert df["cal_date"].tolist() == [20200101, 20200102]
    assert df["pretrade_date"].tolist() == [20191231, 20191231]


def test_fetch_daily_like_table_normalizes_trade_date():
    client = make_client()

    df = client.fetch_table("adj_factor", 20200102)

    assert df["trade_date"].tolist() == [20200102]
