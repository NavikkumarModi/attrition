"""Quick plots over a population-simulation trace.

Takes the plain list-of-dicts trace shape `simulate_population*` return (and
`TraceStore.read()` produces the same shape), so plotting never needs pandas.
matplotlib is soft-imported: everything else in the package works without it,
matching the existing optional `figures` extra in `pyproject.toml`.
"""

__all__ = ["plot_system_value_over_time", "plot_burden_over_time"]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting requires matplotlib: pip install attrition[figures]"
        ) from exc
    return plt


def _cumulative_by_t(trace, key):
    by_t = {}
    for row in trace:
        by_t[row["t"]] = by_t.get(row["t"], 0.0) + row[key]
    ts = sorted(by_t)
    cum, total = [], 0.0
    for t in ts:
        total += by_t[t]
        cum.append(total)
    return ts, cum


def plot_system_value_over_time(trace, ax=None):
    """Cumulative system value (summed across agents) by round."""
    plt = _require_matplotlib()
    ts, cum = _cumulative_by_t(trace, "value")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(ts, cum, marker="o")
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative system value")
    ax.set_title("System value over time")
    return ax


def plot_burden_over_time(trace, ax=None):
    """Alive-arm count by round, as a proxy for accumulated burden -- burden
    rises exactly as arms are destroyed, so a falling `alive` count and a
    rising burden are the same event.
    """
    plt = _require_matplotlib()
    by_t = {}
    for row in trace:
        by_t[row["t"]] = row["alive"]
    ts = sorted(by_t)
    alive = [by_t[t] for t in ts]
    if ax is None:
        _, ax = plt.subplots()
    ax.step(ts, alive, where="post", marker="o")
    ax.set_xlabel("t")
    ax.set_ylabel("arms alive")
    ax.set_title("Pool depletion over time")
    return ax
