"""Tests for render_dashboard: a self-contained HTML file, no server, no
external dependency, from the plain trace shape simulate_population* and
TraceStore.read() already produce.
"""

import json

from attrition import AgentGraph, render_dashboard

_TRACE = [
    {"t": 0, "agent": "a", "arm": 1, "value": 0.5, "regret": 0.1,
     "destroyed": False, "alive": 3},
    {"t": 0, "agent": "b", "arm": 2, "value": 0.4, "regret": 0.2,
     "destroyed": True, "alive": 3},
    {"t": 1, "agent": "a", "arm": 0, "value": 0.6, "regret": 0.0,
     "destroyed": False, "alive": 2},
]


def test_render_dashboard_writes_valid_html(tmp_path):
    path = tmp_path / "d.html"
    out = render_dashboard(_TRACE, path=str(path), title="My Run")
    assert out == str(path)
    content = path.read_text()
    assert content.startswith("<!doctype html>")
    assert "My Run" in content
    assert "<canvas" in content


def test_render_dashboard_embeds_actual_trace_data(tmp_path):
    path = tmp_path / "d.html"
    render_dashboard(_TRACE, path=str(path))
    content = path.read_text()
    assert '"agent": "a"' in content
    assert '"arm": 2' in content


def test_render_dashboard_handles_empty_trace(tmp_path):
    path = tmp_path / "d.html"
    out = render_dashboard([], path=str(path))
    content = path.read_text()
    assert "<!doctype html>" in content
    assert out == str(path)


def test_render_dashboard_omits_graph_panel_by_default(tmp_path):
    path = tmp_path / "d.html"
    render_dashboard(_TRACE, path=str(path))
    content = path.read_text()
    assert 'id="graph"' not in content
    assert "const GRAPH = null;" in content


def test_render_dashboard_includes_graph_panel_when_given(tmp_path):
    graph = AgentGraph.complete(["a", "b"])
    path = tmp_path / "d.html"
    render_dashboard(_TRACE, path=str(path), graph=graph)
    content = path.read_text()
    assert 'id="graph"' in content
    assert '"agents": ["a", "b"]' in content or '"agents":["a","b"]' in content


def test_render_dashboard_returns_the_output_path(tmp_path):
    path = tmp_path / "sub" / "d.html"
    path.parent.mkdir()
    assert render_dashboard(_TRACE, path=str(path)) == str(path)
