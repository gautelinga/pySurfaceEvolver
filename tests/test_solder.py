"""Brakke's ball grid array series: forces, optimizing parameters, and what
running it turned up (restores keeping pressures, undo keeping parameters,
starts with nothing free)."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse


def _linear(ev):
    h = ev.parameters["height"]
    return lambda x: np.c_[0*x[:, :2], x[:, 2]/h]


def test_force_by_linear_move_matches_brakkes_variational_integral(load):
    ev = load("bga-4.fe")
    ev.relax(levels=2)
    fixed = -ev.derivative("height", method="fixed", move=_linear(ev))
    newton = -ev.derivative("height", move=_linear(ev))
    variational = ev.quantity("zforce7").value
    assert fixed == pytest.approx(variational, rel=1e-7)
    assert newton == pytest.approx(variational, rel=1e-5)
    ev.command('read "zforce.cmd"')
    assert float(ev.command("zforce4").split()[1]) == pytest.approx(fixed, rel=1e-7)


def test_derivative_leaves_the_surface_as_it_was(load):
    ev = load("bga-3.fe")
    ev.relax(levels=1)
    x, e, p = ev.vertices, ev.total_energy, ev.bodies().pressure
    for method in ("newton", "fixed"):
        ev.derivative("height", method=method, move=_linear(ev))
        assert np.array_equal(ev.vertices, x) and ev.total_energy == e
        assert np.array_equal(ev.bodies().pressure, p)
    with pytest.raises(ValueError, match="needs move"):
        ev.derivative("height", method="fixed")


def test_a_restored_surface_with_an_energy_quantity_keeps_its_pressures(load):
    # Evolver reset its pressures for every named quantity without a
    # multiplier while reading a datafile (the pad load is an energy), so
    # after any restore Newton refused to run ("Pressures invalid")
    ev = load("bga-5.fe")
    ev.refine(1)
    ev.command("g 5")
    ev.restore(ev.save())
    ev.command("hessian")


def test_relax_balances_an_optimizing_parameter(load):
    ev = load("bga-5.fe")
    assert ev.parameters.optimizing == {"height"}
    r = ev.relax(levels=2)
    assert r.converged and r.stable and r.health.ok
    assert 0.100 < ev.parameters["height"] < 0.110      # 200 dyn carried
    ev.parameters.optimize("height", False)
    assert not ev.parameters.optimizing
    ev.parameters.optimize("height")
    assert ev.parameters.optimizing == {"height"}


def test_undoing_steps_puts_optimizing_parameters_back(load):
    ev = load("bga-5.fe")
    ev.relax(levels=1)
    before = ev._coords()
    h = ev.parameters["height"]
    ev.command("g 20")
    assert ev.parameters["height"] != h
    assert ev._put_back(before)
    assert ev.parameters["height"] == h


def test_a_pad_pulls_itself_back_into_line(load):
    ev = load("bga-9.fe")
    assert ev.parameters["offset"] == pytest.approx(0.03)
    r = ev.relax(levels=2)
    assert r.converged and abs(ev.parameters["offset"]) < 1e-4


def test_relax_refines_a_start_with_nothing_free(load):
    ev = load("bga-30.fe")
    r = ev.relax(levels=2)
    assert r.converged
    with pytest.raises(pyse.EvolverError):
        load("bga-30.fe").relax()          # nothing to refine into


def test_pad_rims_are_not_walls_to_health(load):
    ev = load("bga-3.fe")
    assert ev.relax(levels=2).health.ok


def test_relax_passes_on_each_kind_of_evolver_warning_once(load):
    ev = load("bga-6.fe")                   # overloaded: the pad runs off
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = ev.relax(levels=1)
    assert not r.converged
    evolver = [w for w in caught if issubclass(w.category, pyse.EvolverWarning)]
    codes = [str(w.message).split(":")[0] for w in evolver]
    assert len(codes) == len(set(codes)) and any("more like it" in str(w.message)
                                                 for w in evolver)
