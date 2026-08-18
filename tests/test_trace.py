"""Tests for TraceStore (sqlite3-backed trace persistence)."""

import pytest

from attrition import TraceStore

_ROWS = [
    {"t": 0, "agent": "a", "arm": 1, "value": 0.5, "regret": 0.1,
     "destroyed": False, "alive": 3},
    {"t": 0, "agent": "b", "arm": 2, "value": 0.4, "regret": 0.2,
     "destroyed": True, "alive": 3},
    {"t": 1, "agent": "a", "arm": 0, "value": 0.6, "regret": 0.0,
     "destroyed": False, "alive": 2},
]


def test_write_and_read_roundtrip(tmp_path):
    store = TraceStore(str(tmp_path / "trace.sqlite3"))
    store.write_rows("run-1", _ROWS)
    rows = store.read("run-1")
    assert len(rows) == len(_ROWS)
    assert {r["agent"] for r in rows} == {"a", "b"}
    assert rows[0]["destroyed"] in (0, 1)
    store.close()


def test_multiple_runs_are_isolated(tmp_path):
    store = TraceStore(str(tmp_path / "trace.sqlite3"))
    store.write_rows("run-1", _ROWS[:1])
    store.write_rows("run-2", _ROWS[1:])
    assert len(store.read("run-1")) == 1
    assert len(store.read("run-2")) == 2
    assert len(store.read()) == 3
    assert set(store.run_ids()) == {"run-1", "run-2"}
    store.close()


def test_write_rows_handles_empty_list(tmp_path):
    store = TraceStore(str(tmp_path / "trace.sqlite3"))
    store.write_rows("run-1", [])
    assert store.read("run-1") == []
    store.close()


def test_context_manager_closes(tmp_path):
    with TraceStore(str(tmp_path / "trace.sqlite3")) as store:
        store.write_rows("run-1", _ROWS)
        assert len(store.read()) == len(_ROWS)


def test_to_dataframe_requires_pandas_or_returns_dataframe(tmp_path):
    pd = pytest.importorskip("pandas")
    store = TraceStore(str(tmp_path / "trace.sqlite3"))
    store.write_rows("run-1", _ROWS)
    df = store.to_dataframe("run-1")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(_ROWS)
    store.close()
