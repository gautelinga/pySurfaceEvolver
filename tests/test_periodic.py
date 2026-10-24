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
    kw = dict(axis=(0, 0, 1)) if gauge == "axis" else dict(wet_poles=("south",))
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
                        ({"ball": C.sphere((0, 0, 1), 0.5, gauge="flux")}, "flux")]:
        with pytest.raises(ValueError, match=match):
            pyse.make_datafile(v, [[0, 1, 2]], constraints=cons, periods=PERIODS)
