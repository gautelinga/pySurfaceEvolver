"""recipes.arclength: following a branch of equilibria through a fold."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes


def _wire_catenoid(levels):
    """A soap film between round wires of radius 1, at z = 0 and z = ht."""
    tube = shapes.tube(1.0, 0.0, 0.6, nz=4, nt=16)
    top = C.curve("cos(p1)", "sin(p1)", "ht", lambda x: np.arctan2(x[:, 1], x[:, 0]))
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces, boundaries={1: C.circle(), 2: top},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   parameters={"ht": 0.6})
    ev.relax(levels=levels)
    return ev


def _catenoid_areas(ht):
    """The exact areas of the stable and the unstable catenoid between the
    wires (a*cosh(H/a) = 1, H = ht/2), or None past the fold."""
    H = ht/2
    a_fold = H/1.19967864           # H/a at the fold
    if H > 0.66274:
        return None
    f = lambda a: a*np.cosh(H/a) - 1
    areas = []
    for lo, hi in ((a_fold, 1.0), (1e-3, a_fold)):
        for _ in range(200):
            mid = (lo + hi)/2
            lo, hi = (mid, hi) if (f(lo) > 0) == (f(mid) > 0) else (lo, mid)
        areas.append(np.pi*mid*(2*H + mid*np.sinh(2*H/mid)))
    return areas


def test_around_the_fold_and_down_the_unstable_branch():
    ev = _wire_catenoid(2)
    ev.parameters["ht"] = 0.8
    steps = list(recipes.arclength(ev, "ht", 0.05, 30))
    values = np.array([s.value for s in steps])
    negative = np.array([s.result.negative_modes for s in steps])
    turn = int(np.argmax(values))
    # the turn at the fold (1.3255 for the smooth catenoid), stable before it,
    # one negative mode after
    assert 1.31 < values[turn] < 1.34
    assert (negative[:turn] == 0).all() and (negative[turn + 1:] == 1).all()
    assert (np.diff(values[turn + 1:]) < 0).all() and values[-1] < 0.8
    for s in steps:
        exact = _catenoid_areas(s.value)
        if exact is None:
            continue
        stable, unstable = exact
        expected = stable if s.result.negative_modes == 0 else unstable
        assert s.energy == pytest.approx(expected, rel=2e-3)
        assert s.result.converged and s.result.residual < 1e-8
    assert ev.parameters["ht"] == pytest.approx(values[-1])


def test_the_unstable_catenoids_mode_lowers_the_area():
    ev = _wire_catenoid(1)
    ev.parameters["ht"] = 1.0
    for s in recipes.arclength(ev, "ht", 0.05, 40):
        if s.result.negative_modes == 1:
            break
    m = ev.modes(2)
    assert m.values[0] < 0 < m.values[1]
    # a simple eigenvalue: the same mode again, sign included
    assert np.allclose(ev.modes(1).vectors[0], m.vectors[0], atol=1e-6)
    x, area = ev.vertices, ev.total_area
    for eps in (1e-3, -1e-3):
        ev.vertices = x + eps*m.vectors[0]
        assert ev.total_area < area
    # the neck widens or narrows: the same motion at both ends of the axis
    d = m.vectors[0]
    r = np.hypot(x[:, 0], x[:, 1])
    out = (d[:, 0]*x[:, 0] + d[:, 1]*x[:, 1])/np.maximum(r, 1e-12)
    mid = abs(x[:, 2] - ev.parameters["ht"]/2) < 0.1
    assert abs(out[mid].mean()) > 0.5


def test_bounds_end_the_run_at_the_last_point_inside():
    ev = _wire_catenoid(1)
    ev.parameters["ht"] = 0.8
    steps = list(recipes.arclength(ev, "ht", 0.1, 50, bounds=(0.7, 1.05)))
    assert all(0.7 <= s.value <= 1.05 for s in steps) and len(steps) < 50
    assert ev.parameters["ht"] == pytest.approx(steps[-1].value)
    assert ev.total_area == pytest.approx(steps[-1].energy, rel=1e-9)


def test_a_body_volume_as_the_value(cube):
    cube.refine()
    steps = list(recipes.arclength(cube, cube.body(1), 0.1, 5))
    assert [round(s.value, 6) for s in steps][:2] == [1.0, 1.1]
    assert np.all(np.diff([s.value for s in steps]) > 0)
    for s in steps:
        assert s.volumes[0] == pytest.approx(s.value, rel=1e-8)
    # the energy of a sphere-like surface grows as the volume^(2/3)
    e = np.array([s.energy for s in steps]); v = np.array([s.value for s in steps])
    assert np.allclose(e/v**(2/3), e[0], rtol=2e-2)


def test_newton_target_does_not_depend_on_the_last_one(cube):
    # the corrector differences two Newton steps from one point; the second
    # had started from the pressures the first left
    cube.refine()
    cube.relax()
    put = lambda v: setattr(cube.body(1), "target", v)
    cube.vertices = cube.vertices*1.05
    first, p_first = recipes._newton_target(cube, put, 1.0)
    again, p_again = recipes._newton_target(cube, put, 1.0)
    # (2e-8 and 2e-5 apart before)
    assert np.allclose(first, again, rtol=0, atol=1e-12)
    assert np.allclose(p_first, p_again, rtol=0, atol=1e-12)


def test_a_first_step_too_long_is_halved():
    # past the fold (1.3255) there is nothing at 1.4: the first step halves
    # to 0.3 as later steps do, from where the attempt left the surface
    ev = _wire_catenoid(1)
    ev.parameters["ht"] = 0.8
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        steps = list(recipes.arclength(ev, "ht", 0.6, 2))
    assert [s.value for s in steps] == pytest.approx([0.8, 1.1])
    assert ev.parameters["ht"] == pytest.approx(1.1)


def test_arclength_arguments(cube):
    with pytest.raises(ValueError):
        next(recipes.arclength(cube, cube.body(1), 0.0, 5))
    with pytest.raises(TypeError):
        next(recipes.arclength(cube, lambda ev, v: None, 0.1, 5))
