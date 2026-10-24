"""Slabs periodic in x and y (make_datafile(periods=), _periodic.py)."""

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C
from pysurfaceevolver import shapes

PERIODS = (1.9, 2.3)


def _into_cell(v, periods):
    if periods is None:
        return v
    L = np.asarray(periods)
    v = v.copy()
    v[:, :2] -= L*np.floor(v[:, :2]/L)
    return v


def _drop(periods, angle=60.0):
    """A sessile drop on a floor, centred on the cell's corner: its vertices
    wrapped into the cell, edges across all four sides."""
    cap = shapes.cap(0.5, 0.3, nr=6, nt=24)
    v = cap.vertices.copy()
    rim = np.asarray(cap.rims["rim"])
    v[rim, 2] = 0.0
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), cap.faces,
                   constraints={"bottom": C.plane((0, 0, 1), 0.0, contact_angle=angle)},
                   vertex_constraints={"bottom": rim}, bodies=[pyse.Body(volume=None)],
                   periods=periods)
    ev.body(1).target = ev.body(1).volume
    return ev


def _bead(periods, gauge="axis", zf=0.0):
    """A liquid bridge from a floor (at zf) to a bead above it, on the cell's
    corner."""
    R, c = 0.5, np.array([0.0, 0.0, 0.55 + zf])
    zs = np.linspace(0, 0.35, 6)
    sh = shapes.revolve(np.linspace(0.5, np.sqrt(R*R - 0.2**2), 6), zs + zf, nt=24)
    v = sh.vertices.copy()
    top, bot = np.asarray(sh.rims["top"]), np.asarray(sh.rims["bottom"])
    d = v[top] - c
    v[top] = c + R*d/np.linalg.norm(d, axis=1)[:, None]
    kw = (dict(axis=(0, 0, 1)) if gauge == "axis" else dict(gauge="flux") if gauge == "flux"
          else dict(wet_poles=("south",)))
    cons = {"bottom": C.plane((0, 0, 1), zf, contact_angle=50.0),
            "bead": C.sphere(tuple(c), R, contact_angle=40.0, **kw)}
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), sh.faces, constraints=cons,
                   vertex_constraints={"bottom": bot, "bead": top},
                   bodies=[pyse.Body(volume=None)], periods=periods)
    ev.body(1).target = ev.body(1).volume
    return ev


def _run(make, commands):
    out = []
    for periods in (None, PERIODS):
        ev = make(periods)
        found = [(ev.energy, ev.body(1).volume)]
        for c in commands:
            ev.command(c)
            found.append((ev.energy, ev.body(1).volume))
        out.append((np.array(found), ev.residual(), ev))
    return out


def test_a_drop_across_the_cells_corner_is_the_drop_without_periods():
    # (the floor's wetting a facet integral of z-hat, the volume of z: the
    # same values as the line integrals, step by step, refined, by Newton)
    (plain, _, _), (periodic, res, ev) = _run(
        _drop, ["g 1", "g 5", "r", "g 10"] + ["hessian"]*4)
    assert ev.torus and ev.periods == PERIODS and ev.mesh().periods == PERIODS
    assert np.abs(plain - periodic).max() < 1e-13
    assert res < 1e-9


@pytest.mark.parametrize("gauge, zf", [("axis", 0.0), ("axis", 0.3), ("pole", 0.0)])
def test_a_bridge_to_a_bead_across_the_cells_corner(gauge, zf):
    # (the floor's wetted area under the bead: z-hat's flux through its cap,
    # the bead's line integral; a floor off z = 0, the volume's gauge z - zf,
    # with zf z-hat's flux through the cap)
    (plain, _, _), (periodic, res, ev) = _run(
        lambda p: _bead(p, gauge, zf), ["g 30", "r", "g 60"] + ["hessian"]*4)
    assert np.abs(plain - periodic).max() < (1e-13 if gauge == "axis" else 1e-7)
    assert res < 1e-9
    m = ev.mesh()
    assert np.isclose(m.facet_areas().sum(), float(ev.command("print total_area")), rtol=1e-12)
    assert m.edge_lengths().max() < 0.2


def test_edges_take_the_nearest_image():
    from pysurfaceevolver import _periodic
    x = np.array([[0.05, 0.1, 0.0], [1.85, 2.25, 0.0], [0.1, 0.4, 0.0], [1.0, 0.1, 0.0]])
    w = _periodic.wraps([0, 1], [1, 2], x, PERIODS)
    assert w.tolist() == [[-1, -1], [1, 1]]
    assert np.allclose(_periodic.delta(x[0], x[1], PERIODS), [-0.1, -0.15, 0.0])
    with pytest.raises(ValueError, match="half a period"):
        _periodic.wraps([0], [3], x, PERIODS)


def test_walls_a_slab_cant_have():
    v = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.0]])
    for cons, match in [({"wall": C.plane((1, 0, 0), 0.0)}, "floor"),
                        ({"low": C.plane((0, 0, 1), 0.0), "high": C.plane((0, 0, 1), 1.0)},
                         "one floor"),
                        ({"ball": C.sphere((0, 0, 1), 0.5, gauge="flux"),
                          "bead": C.sphere((0, 0, 3), 0.5, axis=(0, 0, 1))}, "mixed")]:
        with pytest.raises(ValueError, match=match):
            pyse.make_datafile(v, [[0, 1, 2]], constraints=cons, periods=PERIODS)


def test_a_vertex_moved_to_another_image_keeps_the_surface():
    from pysurfaceevolver import _periodic
    ev = _drop(PERIODS)
    e, v = ev.energy, ev.body(1).volume
    m = ev.mesh()
    x = m.vertices[5].copy()
    _periodic.move_near(ev, [int(m.vertex_ids[5])], [x + [PERIODS[0], -PERIODS[1], 0]])
    ev.command("recalc")
    m = ev.mesh()
    assert np.allclose(m.vertices[5] - x, [PERIODS[0], -PERIODS[1], 0])
    assert abs(ev.energy - e) < 1e-14 and abs(ev.body(1).volume - v) < 1e-14
    assert m.edge_lengths().max() < 0.2


def _bead_flux(periods):
    return _bead(periods, gauge="flux")


def test_the_flux_gauges_fields_are_ewalds_sum():
    # (the regions' near sources and polynomials against Ewald's sum over the
    # lattice, at points around the bead and its mirror; derivatives against
    # differences)
    from pysurfaceevolver import _core
    _bead(PERIODS, gauge="flux")
    rng = np.random.default_rng(0)
    for x in rng.uniform([0, 0, 0.05], [PERIODS[0], PERIODS[1], 1.5], (30, 3)):
        if min(np.linalg.norm(x - [a, b, 0.55]) for a in (0, PERIODS[0]) for b in (0, PERIODS[1])) < 0.5:
            continue
        o = np.array(_core.slabflux_probe(list(x)))
        n = len(o)//2
        assert np.abs(o[:n] - o[n:]).max() < 1e-9*np.abs(o[n:]).max()
    for kind, x in [(0, [0.6, 0.7, 0.3]), (1, [0.6, 0.7, 0.3]), (2, [0.1, 0.2, 0.1]),
                    (3, [0.1, 0.2, 0.1])]:
        f = lambda y: np.array(_core.slabflux_field(kind, 0, list(y)))
        o, e = f(x), 1e-5
        d1 = np.array([(f(x + e*d)[:3] - f(x - e*d)[:3])/(2*e) for d in np.eye(3)]).T
        d2 = np.array([(f(x + e*d)[3:12] - f(x - e*d)[3:12]).reshape(3, 3)/(2*e)
                       for d in np.eye(3)]).transpose(1, 2, 0)
        assert np.abs(o[3:12].reshape(3, 3) - d1).max() < 1e-8
        assert np.abs(o[12:].reshape(3, 3, 3) - d2).max() < 1e-7


def test_the_periodic_flux_gauge_is_the_flux_gauge_to_its_quadrature():
    # (the images' and the mirror's fields through the facets by Gauss
    # points: the same surface's values apart by h^4, 16 times less a
    # refinement)
    gaps = []
    for refine in (0, 1, 2):
        found = []
        for periods in (None, PERIODS):
            ev = _bead_flux(periods)
            for _ in range(refine):
                ev.command("r")
            found.append(np.array([ev.energy, ev.body(1).volume]))
        gaps.append(np.abs(found[0] - found[1]))
    assert gaps[0].max() < 1e-6
    assert np.all(gaps[1] < gaps[0]/10) and np.all(gaps[2] < gaps[1]/10)


@pytest.mark.parametrize("gauge", ["axis", "flux"])
def test_newton_steps_far_from_equilibrium_follow_the_lines(gauge):
    # (the Newton step's line following, once off under any symmetry: the
    # slab's steps went their own way, and diverged)
    found = []
    for periods in (None, PERIODS):
        ev = _bead(periods, gauge=gauge)
        for c in ["g 1"]*3 + ["hessian"]*5:
            ev.command(c)
        found.append((ev.energy, ev.residual()))
    assert found[1][1] < 1e-8
    assert abs(found[0][0] - found[1][0]) < (1e-12 if gauge == "axis" else 1e-6)
