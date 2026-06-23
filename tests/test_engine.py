"""Smoke tests for the vaccine_ethics simulation engine.

Research code is validated by reproducible figures, not exhaustive unit tests
(see HOUSEKEEPING.md). These are deliberately lightweight: they confirm the
engine imports cleanly, a small simulation runs end to end, and the report
vector has the documented shape and sane values. Keep grids/agents/iterations
small so the suite stays fast.
"""
import numpy as np
import pytest

from agent_class import FullAgent, FullAgent2
from simulation_class import Simulation, Simulation2, Simulation3


SMALL = dict(grid_size=8, num_agents=40, plot=False)


def _run_small(**overrides):
    params = {**SMALL, **overrides}
    sim = Simulation(**params)
    sim.run(iterations=15, plot_grid=False)
    return sim


def test_engine_imports():
    """The classes the notebooks import are all importable from src/."""
    assert FullAgent is not None
    assert FullAgent2 is not None
    assert all(c is not None for c in (Simulation, Simulation2, Simulation3))


def test_simulation_runs_and_reports():
    """A small single-grid run produces a 14-element report vector."""
    sim = _run_small()
    report = sim.generate_simulation_report()
    assert isinstance(report, np.ndarray)
    assert report.shape == (14,)
    assert np.all(np.isfinite(report))


def test_report_values_are_sane():
    """Proportions stay within [0, 1]; counts are non-negative."""
    report = _run_small().generate_simulation_report()
    # indices 1,2,6,7 are proportions (deaths, infected, non-vul dead, vul dead)
    for idx in (1, 2, 6, 7):
        assert 0.0 <= report[idx] <= 1.0, f"proportion at index {idx} out of range"
    # indices 9..12 are counts (unique infected, total/vul/non-vul infections)
    for idx in (9, 10, 11, 12):
        assert report[idx] >= 0


@pytest.mark.parametrize("strategy", ["vax_all", "vax_vulnerable"])
def test_vaccination_strategies_run(strategy):
    """Both vaccination strategies execute without error and report cleanly."""
    sim = _run_small(**{strategy: True})
    report = sim.generate_simulation_report()
    assert report.shape == (14,)


def test_agents_have_valid_states():
    """Every agent ends in one of the four SEIRD-ish states."""
    sim = _run_small()
    valid = {"S", "I", "R", "D"}
    assert all(agent.state in valid for agent in sim.agents)
