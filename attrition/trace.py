"""Persist population-simulation traces to a queryable store instead of only
returning them in memory -- the piece that lets a run at real scale (long
horizon, many agents) be studied after the fact rather than living only in a
variable that disappears when the script exits.

Uses stdlib `sqlite3` only -- no new required dependency. `.to_dataframe()`
imports `pandas` lazily and only when called, so `pip install attrition` alone
is still enough for everything else in this module.
"""

import sqlite3

__all__ = ["TraceStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace (
    run_id TEXT NOT NULL,
    t INTEGER NOT NULL,
    agent TEXT NOT NULL,
    arm INTEGER NOT NULL,
    value REAL NOT NULL,
    regret REAL NOT NULL,
    destroyed INTEGER NOT NULL,
    alive INTEGER NOT NULL
)
"""


class TraceStore:
    """SQLite-backed sink for `simulate_population`/`simulate_population_simultaneous`
    trace rows.

        store = TraceStore("run.sqlite3")
        simulate_population_simultaneous(pool, population, trace_store=store,
                                         run_id="run-1")
        rows = store.read("run-1")
        df = store.to_dataframe("run-1")   # requires pandas
    """

    def __init__(self, path):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def write_rows(self, run_id, rows):
        """rows: iterable of the dicts `simulate_population*` produces per step."""
        if not rows:
            return
        self._conn.executemany(
            "INSERT INTO trace (run_id, t, agent, arm, value, regret, "
            "destroyed, alive) VALUES (?,?,?,?,?,?,?,?)",
            [(run_id, r["t"], r["agent"], r["arm"], r["value"], r["regret"],
              int(bool(r["destroyed"])), r["alive"]) for r in rows])
        self._conn.commit()

    def read(self, run_id=None):
        """Return matching rows as a list of dicts, ordered by (t, agent)."""
        if run_id is None:
            cur = self._conn.execute("SELECT * FROM trace ORDER BY t, agent")
        else:
            cur = self._conn.execute(
                "SELECT * FROM trace WHERE run_id = ? ORDER BY t, agent",
                (run_id,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def run_ids(self):
        cur = self._conn.execute("SELECT DISTINCT run_id FROM trace")
        return [row[0] for row in cur.fetchall()]

    def to_dataframe(self, run_id=None):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "TraceStore.to_dataframe requires pandas: "
                "pip install attrition[analysis]") from exc
        return pd.DataFrame(self.read(run_id))

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
