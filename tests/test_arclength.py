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


def test_a_body_volume_in_the_lagrange_model(cube):
    # tidying between points vertex-averages only: Evolver can't flip edges
    # in the Lagrange model ("Cannot equiangulate", the first point failed)
    cube.refine()
    cube.relax()
    cube.command("lagrange 2")
    steps = list(recipes.arclength(cube, cube.body(1), 0.1, 4))
    v = np.array([s.value for s in steps])
    assert len(steps) == 4 and np.all(np.diff(v) > 0)
    e = np.array([s.energy for s in steps])
    assert np.allclose(e/v**(2/3), e[0]/v[0]**(2/3), rtol=2e-2)
    assert all(s.result.residual < 1e-8 for s in steps)


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


def test_a_point_that_doesnt_settle_is_tried_again_closer():
    # steps of 0.3 on this coarse mesh overshoot the fold: the corrector's
    # point doesn't settle after tidying (the run stopped at 1.1, then); the
    # step is halved instead, and the run turns at the fold
    ev = _wire_catenoid(1)
    ev.parameters["ht"] = 0.8
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        steps = list(recipes.arclength(ev, "ht", 0.3, 6))
    values = [s.value for s in steps]
    turn = int(np.argmax(values))
    assert 1.31 < values[turn] < 1.34 and turn < len(steps) - 1
    for s in steps:
        exact = _catenoid_areas(s.value)
        expected = exact[0] if s.result.negative_modes == 0 else exact[1]
        assert s.energy == pytest.approx(expected, rel=2e-3)
        assert s.result.residual < 1e-8
    assert steps[-1].result.negative_modes == 1


def test_a_sharp_fold_is_turned_along_the_tangent():
    # a drop hanging from a ring of radius 1 (Bond 2), grown in its volume:
    # at the largest volume the branch turns almost at a right angle in the
    # step's measure, and the secant of the last two points pointed off it
    # whatever the step (stopped at 3.03, with steps of 0.04 down to 0.005)
    h = np.real(np.roots([1.0, 0.0, 3.0, -3.0])).max()     # the cap of volume pi/2
    cap = shapes.cap((1 + h*h)/(2*h), h, nr=6, nt=32)
    ev = pyse.Evolver()
    ev.load_arrays(cap.vertices*[1, 1, -1], cap.faces[:, ::-1],
                   boundaries={1: C.circle()}, vertex_boundaries={1: cap.rims["rim"]},
                   gravity=2.0, bodies=[pyse.Body(volume=np.pi/2, density=1.0)])
    ev.relax()
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for s in recipes.arclength(ev, ev.body(1), 0.04, 80):
            values.append(s.value)
            if len(values) > 3 and values[-1] < values[-2] < values[-3]:
                break
    assert 3.0 < max(values) < 3.1 and values[-1] < max(values) - 0.02
    assert s.result.negative_modes == 1


def _disk_bridge(height, bond=0.0):
    """A bridge between two disks of radius 1 a ``height`` apart, pinned on
    their edges (the upper disk's volume by its wire's content integral),
    holding the cylinder's volume, relaxed once refined."""
    tube = shapes.tube(1.0, 0.0, height, nz=max(2, round(height*24/(2*np.pi))), nt=24)
    ev = pyse.Evolver()
    top = C.circle((0, 0, height), content=(f"-{height!r}*y/2", f"{height!r}*x/2", "0"))
    ev.load_arrays(tube.vertices, tube.faces, boundaries={1: C.circle(), 2: top},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=np.pi*height, density=1.0 if bond else None)],
                   gravity=bond if bond else None)
    ev.relax(levels=1)
    return ev


def _to_least_volume(ev, ds, **kwargs):
    """Take liquid away with arclength (warnings as errors) until the volume
    has risen for three points; the reduced volumes and the last step."""
    cylinder = ev.body(1).target
    values = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("ignore", pyse.EvolverWarning)
        for s in recipes.arclength(ev, ev.body(1), ds*cylinder, 80, **kwargs):
            values.append(s.value/cylinder)
            if len(values) > 3 and values[-1] > values[-2] > values[-3]:
                break
    return values, s


def test_a_quadratic_surface_is_not_spread_between_points():
    # taking liquid from the bridge to its least volume (pi*0.2544, where it
    # pinches): spreading the vertices between points left the quadratic
    # surface a little worse each time (Newton steps don't move nodes along
    # it), the run went 2.3% past the exact fold and stopped there
    ev = _disk_bridge(1.0)
    ev.command("quadratic")
    values, s = _to_least_volume(ev, -0.04, halvings=6)
    assert min(values) == pytest.approx(0.2544, rel=5e-3)
    assert s.result.negative_modes == 1


def test_the_default_halvings_get_round_a_fold_with_gravity():
    # a bridge between disks 3.92 apart (radius 1, Bond 0.2) at its least
    # volume, 0.7730 of the cylinder's (Slobozhanin and Perales): the step
    # into the fold needed five halvings, and the default was three
    values, s = _to_least_volume(_disk_bridge(3.92, bond=0.2), -0.04)
    assert min(values) == pytest.approx(0.7730, rel=5e-3)
    assert s.result.negative_modes == 1


def test_remeshing_keeps_the_unstable_branch():
    # the catenoid stretches to its fold and its neck then thins: without
    # remeshing the facets degrade and the run loses the unstable branch
    # (back to no negative mode at 0.85, areas 30% off); remeshing between
    # points keeps it, down to wires 0.5 apart (where the neck is a ring of
    # four edges: events would stop the run before, at five)
    ev = _wire_catenoid(1)
    ev.parameters["ht"] = 0.8
    h = float(np.median(ev.mesh().edge_lengths()))
    n = len(ev.vertices)
    steps = list(recipes.arclength(ev, "ht", 0.05, 33, remesh=h, events=False))
    values = np.array([s.value for s in steps])
    negative = np.array([s.result.negative_modes for s in steps])
    turn = int(np.argmax(values))
    assert 1.31 < values[turn] < 1.34 and values[-1] < 0.55
    assert (negative[:turn] == 0).all() and (negative[turn + 1:] == 1).all()
    for s in steps:
        exact = _catenoid_areas(s.value)
        if exact is not None:
            expected = exact[0] if s.result.negative_modes == 0 else exact[1]
            assert s.energy == pytest.approx(expected, rel=6e-3)
    assert len(ev.vertices) > 1.5*n
    length = ev.mesh().edge_lengths()
    assert length.max() < 1.6*h and ev.mesh_quality().angle_min > 15


def test_remeshing_by_a_function():
    ev = _wire_catenoid(1)
    ev.parameters["ht"] = 0.8
    h = float(np.median(ev.mesh().edge_lengths()))
    calls = []

    def remesh(ev):
        calls.append(ev.parameters["ht"])
        ev.remesh(target=h)
    steps = list(recipes.arclength(ev, "ht", 0.05, 8, remesh=remesh))
    assert len(calls) == 6 and len(ev.vertices) > 288
    assert all(s.result.residual < 1e-8 for s in steps)


def test_carry_a_field_to_a_refined_mesh():
    ev = _wire_catenoid(0)
    ids, x = ev.mesh().vertex_ids, ev.vertices
    ev.refine()
    m = ev.mesh()
    carried = recipes._carry(ids, x, m)
    kept = np.isin(m.vertex_ids, ids)
    assert np.array_equal(carried[kept], x[np.searchsorted(ids, m.vertex_ids[kept])])
    # a new vertex (an edge's midpoint) takes its neighbours' mean: near it
    gap = np.linalg.norm(carried[~kept] - m.vertices[~kept], axis=1)
    assert (~kept).sum() > 0 and gap.max() < np.median(m.edge_lengths())


def test_arclength_arguments(cube):
    with pytest.raises(ValueError):
        next(recipes.arclength(cube, cube.body(1), 0.0, 5))
    with pytest.raises(TypeError):
        next(recipes.arclength(cube, lambda ev, v: None, 0.1, 5))


def test_the_correctors_two_newton_steps_come_from_one_factorization():
    # (at a body's target and at that plus delta: only the constraint rows of
    # the right side change, so one factorization gives both, the same)
    drop = shapes.cap(0.7, 0.7, nr=6, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(drop.vertices, drop.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
                   vertex_constraints={1: drop.rims["rim"]}, bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    ev.relax()
    body = ev.body(1)

    def put(v):
        body.target = v
        ev.command("recalc")
    put.body = 1
    value, delta = 1.1*body.target, 1e-6
    start = ev._coords()
    n0, m0 = recipes._newton_target(ev, put, value)
    n1, m1 = recipes._newton_target(ev, put, value + delta)
    ev._put_back(start)
    x0, q0, x1, q1 = recipes._newton_pair(ev, put, value, delta)
    assert np.allclose(x0, n0, rtol=0, atol=1e-12) and np.allclose(x1, n1, rtol=0, atol=1e-12)
    assert np.allclose(q0, m0, rtol=1e-12) and np.allclose(q1, m1, rtol=1e-12)
    assert np.array_equal(ev.vertices, start.coords)
