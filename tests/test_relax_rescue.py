"""relax()'s rescue branches, which no natural case reaches: forced through
the engine calls they react to."""

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import EvolverError, _core


def test_non_finite_energy_puts_the_round_back(cube, monkeypatch):
    cube.relax(stability=False)
    cube.body(1).target = 1.3
    start = cube.vertices
    real = _core.total_energy
    calls = []

    def energy():
        calls.append(1)
        return float("nan") if len(calls) == 2 else real()   # the first step

    monkeypatch.setattr(_core, "total_energy", energy)
    with pytest.raises(EvolverError, match="energy became nan"):
        cube.relax(stability=False)
    assert np.array_equal(cube.vertices, start)    # the round's start: the first one


def test_failed_newton_first_attempt_is_skipped_next_call(cube, monkeypatch):
    # Lagrange model: five gradient steps and Newton first; when that fails
    # the surface goes back, the usual rounds follow, and the next call skips
    # the attempt (then tries it again)
    cube.relax(stability=False)
    cube.set_model("lagrange", 2)
    real = pyse.Evolver.command
    tried = []

    def command(self, text):
        if text in ("hessian", "hessian_seek"):
            tried.append(text)
            raise EvolverError("no Hessian today")
        return real(self, text)

    monkeypatch.setattr(pyse.Evolver, "command", command)
    counts = []
    for target in (1.1, 1.2, 1.3):
        cube.body(1).target = target
        tried.clear()
        r = cube.relax(stability=False, max_iter=30)
        assert r.newton_steps == 0
        counts.append(len(tried))
    # attempt + the level's Newton, then the level's Newton only, then both
    assert counts == [4, 2, 4]


def test_relax_result_repr(cube):
    r = cube.relax()
    text = repr(r)
    assert text.startswith("IterationResult(converged after ")
    assert "Newton" in text and "stable" in text and "health ok" in text
    assert repr(cube.iterate(2)).startswith("IterationResult(2 gradient steps, energy ")
