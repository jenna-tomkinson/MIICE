"""Shared pytest fixtures for the miice test suite."""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _no_plot_windows(monkeypatch):
    """Prevent tests from blocking on interactive plot windows and close
    any figures created during the test."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")
