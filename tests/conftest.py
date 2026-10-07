import pathlib
import subprocess
import sys
import textwrap

import pytest

from pysurfaceevolver import Evolver

ROOT = pathlib.Path(__file__).resolve().parents[1]
FE_DIR = ROOT / "fe"


@pytest.fixture
def fe_dir(monkeypatch):
    """Run the test from fe/, so datafiles find the files they include."""
    monkeypatch.chdir(FE_DIR)
    return FE_DIR


@pytest.fixture
def load(fe_dir):
    """Return a function that loads a sample datafile.

    The Evolver engine is shared by the whole process; every Evolver is a
    handle to it, and loading replaces the surface.
    """
    def _load(name, **kwargs):
        return Evolver(name, **kwargs)
    return _load


@pytest.fixture
def cube(load):
    return load("cube.fe")


@pytest.fixture
def run_python():
    """Run a Python snippet in a separate process, from fe/.

    For tests that need a process of their own, such as sending SIGINT.
    """
    def _run(code, timeout=120):
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True, text=True, timeout=timeout, cwd=FE_DIR,
        )
    return _run
