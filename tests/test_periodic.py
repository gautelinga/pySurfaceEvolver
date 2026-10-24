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


def test_the_datafiles_periods_are_plain_numbers():
    # NumPy 2 writes a scalar's repr as np.float64(...), which Evolver can't read
    from pysurfaceevolver import _periodic
    text = _periodic.header((np.float64(1.9), np.float32(2.5)))
    assert "np" not in text and "1.9 0 0" in text and "0 2.5 0" in text


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


def test_a_bodys_volconst_survives_equiangulation():
    # (Evolver's fix_volconst, at the load and after 'u' flips an edge,
    # rounded every volconst to a multiple of the torus cell's volume: in a
    # slab, Lx Ly 1e6, the walls' closures went; and its volume adjustments
    # put a negative volume a cell's volume up)
    ev = _bead(PERIODS, gauge="pole")
    b = ev.body(1)
    v = b.volconst
    assert v > 0.1                       # (the wet south pole's closure)
    b.volconst = v + 0.25
    ev.vertices = ev.vertices + 0.02*np.random.default_rng(0).normal(size=ev.vertices.shape)*[1, 1, 0]
    assert "switched" in ev.command("u")
    assert abs(b.volconst - (v + 0.25)) < 1e-15


def test_a_walls_gap_is_at_its_nearest_image():
    # (Evolver evaluates a constraint at a vertex's coordinates: a vertex at
    # another image than a sphere's looked a period away from it)
    from pysurfaceevolver import _periodic
    ev = _bead(PERIODS)
    m = ev.mesh()
    free = np.flatnonzero(~ev.on_constraint("bead") & ~ev.on_constraint("bottom"))[:5]

    def gaps():
        _, _, dist, names = ev._constraint_gaps()
        return np.asarray(dist)[free, names.index("bead")]

    before = gaps()
    _periodic.move_near(ev, m.vertex_ids[free], m.vertices[free] + [PERIODS[0], PERIODS[1], 0])
    assert np.allclose(gaps(), before, atol=1e-12) and np.all(np.abs(before) < 0.5)


@pytest.mark.parametrize("make", [_drop, _bead])
def test_adapt_merges_and_splits_as_without_periods(make):
    # (merges were refused under any symmetry: Evolver's shape check took no
    # periods; it takes a slab's at the minimum image now)
    found = []
    for periods in (None, PERIODS):
        ev = make(periods)
        ev.command("r; g 20")
        n = ev.adapt()
        ev.relax()
        found.append((n, len(ev.mesh().facets), ev.energy, ev.body(1).volume))
    assert found[0][:2] == found[1][:2] and found[0][0] > 0
    assert np.allclose(found[0][2:], found[1][2:], rtol=1e-11)


def test_the_flux_gauges_field_far_above_the_beads():
    # (a box in a layer not made yet: the layer's lock was the box's own,
    # taken again inside it, and the step hung)
    from pysurfaceevolver import _core
    _bead(PERIODS, gauge="flux")
    for z in (2.0, 3.5, 5.0):
        o = np.array(_core.slabflux_probe([0.3, 0.4, z]))
        n = len(o)//2
        assert np.abs(o[:n] - o[n:]).max() < 1e-9*np.abs(o[n:]).max()


def _drop_among_spheres(mode, n, floor=False, box=False):
    from pysurfaceevolver import shapes as S
    cap = S.cap(0.5, 0.25, nr=4, nt=12)
    v = cap.vertices + [0, 0, 1.0]
    rim = np.asarray(cap.rims["rim"])
    v[rim] /= np.linalg.norm(v[rim], axis=1)[:, None]
    cons = {"ball": C.sphere((0, 0, 0), 1.0, contact_angle=60.0, gauge="flux")}
    for i in range(n):
        cons[f"s{i}"] = C.sphere((0.6*(i % 9) - 2.4, 0.6*(i//9) - 2.4, -2.0 - 0.1*(i % 3)), 0.2,
                                 contact_angle=60.0, gauge="flux")
    if floor:
        cons["bottom"] = C.plane((0, 0, 1), -3.0, contact_angle=90.0)
    if box:
        for k, (n_, d) in enumerate([((1, 0, 0), -4.0), ((-1, 0, 0), -4.0),
                                     ((0, 1, 0), -4.0), ((0, -1, 0), -4.0)]):
            cons[f"side{k}"] = C.plane(n_, d, contact_angle=90.0)
    ev = pyse.Evolver()
    ev.load_arrays(v, cap.faces, constraints=cons, vertex_constraints={"ball": rim},
                   bodies=[pyse.Body(volume=None)], flux_sum=mode)
    ev.body(1).target = ev.body(1).volume
    return ev


@pytest.mark.parametrize("n", [10, 70])
def test_the_fast_flux_sum_is_the_direct_one(n):
    # (spheres only: the near ones directly, the far ones by local
    # expansions, against every sphere's sum, through steps and Newton)
    found = []
    for mode in ("direct", "fast"):
        ev = _drop_among_spheres(mode, n)
        out = [(ev.energy, ev.body(1).volume)]
        for c in ["g 5", "r", "g 10", "hessian", "hessian"]:
            ev.command(c)
            out.append((ev.energy, ev.body(1).volume))
        found.append(np.array(out))
    assert np.abs(found[0] - found[1]).max() < 1e-11


def test_the_fast_flux_sum_in_a_box():
    # (the spheres' images in the floor and the box's walls, a lattice of
    # twice the box: no line integrals on the walls; the same as the direct
    # sums' to the facets' quadrature of the images' fields)
    found = []
    for mode in ("direct", "fast"):
        ev = _drop_among_spheres(mode, 20, floor=True, box=True)
        ev.command("g 5; r; g 10; hessian; hessian")
        found.append((ev.energy, ev.body(1).volume, ev.residual()))
    assert found[1][2] < 2*found[0][2] + 1e-12
    assert abs(found[0][0] - found[1][0]) < 1e-5 and abs(found[0][1] - found[1][1]) < 1e-5
    with pytest.raises(ValueError, match="fast"):
        pyse.make_datafile(np.eye(3), [[0, 1, 2]], flux_sum="fast", constraints={
            "ball": C.sphere((0, 0, 0), 1.0, gauge="flux"),
            "tilted": C.plane((1, 0, 1), 0.0)})


def _short_bridge(periods):
    """_bead's bridge in one band: every edge from a floor vertex to a bead
    vertex, across the cell's corner."""
    R, c = 0.5, np.array([0.0, 0.0, 0.55])
    sh = shapes.revolve([0.5, np.sqrt(R*R - 0.2**2)], [0.0, 0.35], nt=24)
    v = sh.vertices.copy()
    top, bot = np.asarray(sh.rims["top"]), np.asarray(sh.rims["bottom"])
    d = v[top] - c
    v[top] = c + R*d/np.linalg.norm(d, axis=1)[:, None]
    cons = {"bottom": C.plane((0, 0, 1), 0.0, contact_angle=50.0),
            "bead": C.sphere(tuple(c), R, contact_angle=40.0, axis=(0, 0, 1))}
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), sh.faces, constraints=cons,
                   vertex_constraints={"bottom": bot, "bead": top},
                   bodies=[pyse.Body(volume=None)], periods=periods)
    return ev


def test_deleting_an_edge_keeps_a_spheres_vertices_at_its_centre():
    # (Evolver's torus_unwrap_edge moved the head of a wrapped edge between
    # the floor and a sphere, a sphere's vertex, a period from its centre)
    ev = _short_bridge(PERIODS)
    m = ev.mesh()
    on_bead, on_floor = ev.on_constraint("bead"), ev.on_constraint("bottom")
    x = np.asarray(m.vertices)
    e = np.asarray(m.edges)
    raw = np.linalg.norm(x[e[:, 1], :2] - x[e[:, 0], :2], axis=1)
    pick = np.flatnonzero(on_floor[e[:, 0]] & on_bead[e[:, 1]] & (raw > 1.0))
    assert len(pick)
    ev.command(f"delete edge[{int(m.edge_ids[pick[0]])}]")
    m = ev.mesh()
    x = np.asarray(m.vertices)
    r = np.linalg.norm(x[ev.on_constraint("bead")] - [0.0, 0.0, 0.55], axis=1)
    assert np.allclose(r, 0.5, atol=1e-9)
    assert m.edge_lengths().max() < 0.5


def test_a_vertex_near_two_spheres_goes_to_the_nearer_ones_copy():
    # (_home_by_spheres moved it to each in turn: a period from the first)
    from pysurfaceevolver import _surgery as S
    L = PERIODS
    cap = shapes.cap(0.12, 0.08, nr=3, nt=12)
    v, rim = cap.vertices.copy(), np.asarray(cap.rims["rim"])
    v[rim, 2] = 0.0
    cons = {"bottom": C.plane((0, 0, 1), 0.0, contact_angle=60.0),
            "beadA": C.sphere((0.33, 0.0, 0.4), 0.3, contact_angle=40.0),
            "beadB": C.sphere((L[0] - 0.33, 0.0, 0.4), 0.3, contact_angle=40.0)}
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v + [0.02, 0.0, 0.0], L), cap.faces, constraints=cons,
                   vertex_constraints={"bottom": rim}, bodies=[pyse.Body(volume=None)],
                   periods=L)
    S._home_by_spheres(ev, ["beadA", "beadB"])
    x = np.asarray(ev.mesh().vertices)
    for key in ("beadA", "beadB"):
        c = np.asarray(cons[key].shape[1])
        raw = np.linalg.norm(x - c, axis=1)
        near = np.linalg.norm(_delta(c, x), axis=1)
        other = np.asarray(cons["beadB" if key == "beadA" else "beadA"].shape[1])
        closer = near < np.linalg.norm(_delta(other, x), axis=1)
        # (each vertex at the copy of the sphere it is nearest)
        assert np.allclose(raw[closer & (near < 0.5)], near[closer & (near < 0.5)])


def _delta(c, x):
    from pysurfaceevolver import _periodic
    return _periodic.delta(c, x, PERIODS)


def test_flux_spheres_crossing_the_floor_or_each_other_are_refused():
    # (a source inside another sphere: its flux through the wetted part has
    # no line integral; the mirror of a sphere on the floor sat at its centre:
    # NaN)
    v, f = np.array([[0, 0, 2.0], [0.2, 0, 2.0], [0, 0.2, 2.0]]), [[0, 1, 2]]
    for cons in ({"bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                  "ball": C.sphere((0.5, 0.5, 0.0), 0.5, contact_angle=30.0, gauge="flux")},
                 {"bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                  "ball": C.sphere((0.5, 0.5, 1.0), 0.5, contact_angle=30.0, gauge="flux"),
                  "other": C.sphere((0.7, 0.5, 1.0), 0.5, contact_angle=30.0, gauge="flux")}):
        ev = pyse.Evolver()
        with pytest.raises(ValueError, match="must not cross"):
            ev.load_arrays(v, f, constraints=cons, periods=PERIODS)


def test_the_flux_gauge_far_above_and_beside_a_box():
    # (points outside the grid raised an error inside the integrand, on any
    # thread: the grid now covers far above the beads and, in a box, the
    # whole cell of its images)
    from pysurfaceevolver import _core
    _bead(PERIODS, gauge="flux")
    o = np.array(_core.slabflux_probe([0.3, 0.4, 30.0]))
    n = len(o)//2
    assert np.abs(o[:n] - o[n:]).max() < 1e-9*np.abs(o[n:]).max()
    ev = _drop_among_spheres("fast", 20, floor=True, box=True)
    e = ev.energy
    ev.command("g 3")
    assert np.isfinite(ev.energy) and np.isfinite(e)
    for x in ([-4.2, 0.0, -1.0], [3.9, -4.1, 0.5]):
        assert len(_core.slabflux_probe(x)) > 0


def test_a_plane_given_by_expressions_is_refused_plainly():
    # (plane() takes numbers; a hand-made wall whose normal has an expression
    # got numpy's conversion error)
    import dataclasses
    slope = dataclasses.replace(C.plane((0, 0, 1), 0.0), shape=("plane", ("tilt", 0, 1), 0.0))
    with pytest.raises(ValueError, match="floor"):
        pyse.make_datafile(np.eye(3), [[0, 1, 2]], parameters={"tilt": 0.1},
                           constraints={"slope": slope}, periods=PERIODS)
