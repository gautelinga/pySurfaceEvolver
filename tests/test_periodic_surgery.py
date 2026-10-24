"""Changes of topology in slabs periodic in x and y (_surgery, _relax,
recipes): the same configuration with and without periods, the feature
straddling the cell's sides or corner (its vertices wrapped into the cell),
gives the same results."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import _surgery as S
from pysurfaceevolver import constraints as C, recipes, shapes
from pysurfaceevolver._events import Event

PERIODS = (1.9, 2.3)
WIDE = (3.1, 2.9)
HOUR = (4.5, 4.3)          # (the hourglass's edges are up to 1.7 long)


def _into_cell(v, periods):
    if periods is None:
        return np.asarray(v, float)
    L = np.asarray(periods)
    v = np.array(v, float)
    v[:, :2] -= L*np.floor(v[:, :2]/L)
    return v


def _same_image(a, b, periods, tol=1e-12):
    """Whether points a and b are images of each other."""
    d = np.asarray(b, float) - np.asarray(a, float)
    L = np.asarray(periods)
    d[..., :2] -= L*np.round(d[..., :2]/L)
    return np.abs(d).max() < tol


# ---- the helpers --------------------------------------------------------------


def test_without_periods_the_helpers_are_the_plain_arithmetic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 3))
    f = rng.integers(0, 30, size=(20, 3))
    a, b = x[:10], x[10:20]
    assert np.array_equal(S._delta(a, b, None), b - a)
    assert np.array_equal(S._mid(a, b, None), (a + b)/2)
    assert np.array_equal(S._middle(x[f], None, axis=1), x[f].mean(axis=1))
    assert np.array_equal(S._middle(a, None), a.mean(axis=0))
    assert np.array_equal(S._corners(x, f, None), x[f])
    assert np.array_equal(S._cross(x, f, None),
                          np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]]))
    moved = x.copy()
    moved[[1, 4]] = a[0]
    assert np.array_equal(S._put_at(x, [1, 4], a[0], None), moved)
    assert np.array_equal(recipes._coming([], [], None), [])


def test_the_helpers_take_nearby_points_across_the_corner():
    rng = np.random.default_rng(2)
    plain = rng.normal(scale=0.2, size=(12, 3))          # (round the cell's corner)
    x = _into_cell(plain, PERIODS)
    f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    assert np.allclose(S._delta(x[0], x[1:], PERIODS), plain[1:] - plain[0])
    assert _same_image(S._mid(x[:6], x[6:], PERIODS), (plain[:6] + plain[6:])/2, PERIODS)
    assert _same_image(S._middle(x, PERIODS), plain.mean(axis=0), PERIODS)
    middles = S._middle(x[f], PERIODS, axis=1)
    assert _same_image(middles, plain[f].mean(axis=1), PERIODS)
    c = S._corners(x, f, PERIODS)
    assert np.allclose(c - c[:, :1], plain[f] - plain[f][:, :1])
    near = np.array([PERIODS[0], 0.0, 0.0])
    c = S._corners(x, f, PERIODS, near)
    assert np.allclose(c - near, plain[f])              # (all at the images by near)
    assert np.allclose(S._cross(x, f, PERIODS),
                       np.cross(plain[f[:, 1]] - plain[f[:, 0]], plain[f[:, 2]] - plain[f[:, 0]]))
    # a point written at each vertex's own image of it: within reach of where it was
    moved = S._put_at(x, [0, 5], plain[3], PERIODS)
    assert _same_image(moved[[0, 5]], plain[[3, 3]], PERIODS)
    assert np.abs(moved[[0, 5]] - x[[0, 5]]).max() < 1.0
    assert np.array_equal(np.delete(moved, [0, 5], axis=0), np.delete(x, [0, 5], axis=0))
    # the closest pair of rows and the strip between two of them
    on = np.ones(12, dtype=bool)
    rows = np.arange(12)
    assert S._closest_pair(x, rows, on, PERIODS) == S._closest_pair(plain, rows, on)
    off = ~on
    off[[3, 7]] = True
    assert list(S._strip(x, 0, 1, ~off, 0.1, PERIODS)) == list(S._strip(plain, 0, 1, ~off, 0.1))


def _tube_round_the_cell(n=12, nt=8, r=0.3, z=1.0):
    """A closed tube along x, round the cell: no border, no inside or outside
    in the plane."""
    s = np.arange(n)*PERIODS[0]/n
    t = np.arange(nt)*2*np.pi/nt
    v = np.array([[x, 1.0 + r*np.cos(a), z + r*np.sin(a)] for x in s for a in t])
    f = []
    for i in range(n):
        for j in range(nt):
            a, b = i*nt + j, i*nt + (j + 1) % nt
            c, d = ((i + 1) % n)*nt + j, ((i + 1) % n)*nt + (j + 1) % nt
            f += [[a, c, d], [a, d, b]]
    return v, np.array(f)


def test_a_piece_is_unwrapped_unless_it_goes_round_the_cell():
    t = np.linspace(0, np.pi, 7)
    ball = shapes.revolve(0.3*np.sin(t), -0.3*np.cos(t), 10)
    x = _into_cell(ball.vertices, PERIODS)
    lifted, wraps = S._lifted(x, np.asarray(ball.faces), PERIODS)
    assert not wraps
    shift = lifted[0] - ball.vertices[0]
    assert np.allclose(lifted - shift, ball.vertices)
    v, f = _tube_round_the_cell()
    assert S._lifted(v, f, PERIODS)[1]


def _crumb_and_tube(periods):
    """A cube's body, a closed piece of four facets lying flat on the cell's
    corner (a crumb), and in a slab a tube round the cell."""
    box = shapes.box(n=2)
    v = np.asarray(box.vertices, float)*0.8 + [0.5, 0.2, 2.0]
    f = np.asarray(box.faces)
    flat = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.0, 0.15, 0], [0.0, 0.0, 0]], float)
    tet = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]) + len(v)
    verts, faces = np.r_[v, flat], np.r_[f, tet]
    if periods is not None:
        tv, tf = _tube_round_the_cell()
        verts, faces = np.r_[verts, tv], np.r_[faces, tf + len(verts)]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(verts, periods), faces, bodies=[pyse.Body(faces=range(len(f)))],
                   periods=periods)
    return ev, len(f), len(v)


def test_a_crumb_across_the_corner_goes_and_a_tube_round_the_cell_stays():
    for periods in (None, PERIODS):
        ev, nf, nv = _crumb_and_tube(periods)
        volume = ev.body(1).volume
        tube = 0 if periods is None else 2*12*8
        assert S._drop_empty(ev) == 1
        assert ev.counts["facets"] == nf + tube
        assert ev.body(1).volume == pytest.approx(volume)
        assert S._drop_empty(ev) == 0


def _hourglass(waist, periods):
    """test_topology's hourglass (two triangular double cones joined at a
    waist ``waist`` across, one body) on the cell's corner."""
    def tri(r, z):
        a = np.arange(3)*2*np.pi/3
        return np.c_[r*np.cos(a), r*np.sin(a), np.full(3, z)]
    v = np.r_[[[0, 0, -0.6]], tri(1, 0), tri(waist/np.sqrt(3), 1), tri(1, 2), [[0, 0, 2.6]]]
    f = []
    for i in range(3):
        j = (i + 1) % 3
        f.append([0, 1 + j, 1 + i])
        for lo in (1, 4):
            hi = lo + 3
            f += [[lo + i, lo + j, hi + j], [lo + i, hi + j, hi + i]]
        f.append([10, 7 + i, 7 + j])
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), np.array(f), bodies=[pyse.Body(faces=range(len(f)))],
                   periods=periods)
    ev.body(1).target = ev.body(1).volume
    return ev


def test_a_ring_collapsed_across_the_corner_is_one_vertex(monkeypatch):
    # (_cut: each ring vertex written at its own image of the centre, so the
    # edges' wraps stay right; the separation after it, a C edit, stubbed)
    from pysurfaceevolver import _edits
    monkeypatch.setattr(_edits, "edit", lambda ev, kind, ids, **kw: np.zeros(len(ids), int))
    found = []
    for periods in (None, HOUR):
        ev = _hourglass(0.02, periods)
        m = ev.mesh()
        h = S._median_edge(m)
        (ring,) = S._collapsed_rings(m, h)
        x = np.asarray(m.vertices, float)
        centre = S._middle(x[ring], periods)
        S._cut(ev, x, ring, centre, h)
        m = ev.mesh()
        assert m.edge_lengths().max() < 2.0 and np.isfinite(ev.energy)
        assert ev.counts["facets"] == 12 and ev.counts["vertices"] == 9
        at = np.flatnonzero(np.linalg.norm(m.delta(centre, m.vertices), axis=1) < 1e-6)
        assert len(at) == 1
        found.append((ev.counts["vertices"], ev.counts["edges"], m.facet_areas().sum(),
                      ev.body(1).volume))
    assert found[0][:2] == found[1][:2]
    assert np.allclose(found[0][2:], found[1][2:], rtol=1e-12)


def test_the_ring_round_a_neck_across_the_corner_is_the_ring_without_periods():
    t = np.linspace(0.15, np.pi - 0.15, 15)
    z = -1.2*np.cos(t)
    r = 0.6*np.sin(t)*(0.15 + 0.85*np.cos(t)**2)
    sh = shapes.revolve(r, z, 16)
    f, plain = np.asarray(sh.faces), np.asarray(sh.vertices, float)
    e = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1), axis=0)
    waist = np.flatnonzero(np.abs(plain[:, 2]) < 0.2)
    gap = 2*np.hypot(plain[waist, 0], plain[waist, 1]).min()
    event = Event("neck", gap, 0.1, np.zeros(3), waist)
    ring, centre, side = S._ring(plain, f, e, event)
    assert len(ring) >= 6
    x = _into_cell(plain, PERIODS)
    for where in (np.zeros(3), np.array([PERIODS[0], PERIODS[1], 0.0])):
        moved = Event("neck", gap, 0.1, where, waist)
        r2, c2, s2 = S._ring(x, f, e, moved, periods=PERIODS)
        assert list(r2) == list(ring)
        assert np.allclose(s2, side, atol=1e-12) or np.allclose(s2, -side, atol=1e-12)
        assert _same_image(c2, centre, PERIODS)
        assert S._sheet(x, f, moved, PERIODS) == S._sheet(plain, f, event)


def _two_balls(periods, gap=0.02, R=0.4):
    """Two balls gap apart across the cell's side x = 0."""
    t = np.linspace(0, np.pi, 9)
    ball = shapes.revolve(R*np.sin(t), -R*np.cos(t), 16)
    v = np.asarray(ball.vertices, float) + [0, 1.0, 1.0]
    n, nf = len(v), len(ball.faces)
    verts = np.r_[v - [R + gap/2, 0, 0], v + [R + gap/2, 0, 0]]
    return verts, np.r_[ball.faces, ball.faces + n], nf


def test_the_facets_facing_across_a_touch_on_the_side():
    verts, f, _ = _two_balls(None)
    p = np.array([0.0, 1.0, 1.0])
    want = S._facing_facets(verts, f, p)
    x = _into_cell(verts, PERIODS)
    for where in (p, p + [PERIODS[0], 0, 0]):
        assert S._facing_facets(x, f, where, periods=PERIODS) == want


def test_events_a_period_apart_are_at_one_place():
    # (events are compared at the nearest images: one place, a period apart)
    e1 = Event("contact lines", 0.10, 0.1, np.array([0.01, 0.5, 0.0]), np.array([0]), 1)
    e2 = Event("contact lines", 0.05, 0.1, np.array([PERIODS[0] - 0.01, 0.5, 0.0]),
               np.array([0]), 1)
    assert recipes._coming([e2], [e1]) == []
    assert recipes._coming([e2], [e1], PERIODS) == [e2]
    a = S._Tried("solid", frozenset(["c1"]), e1.where, 0.1, 0.5, 1, "made")
    b = S._Tried("solid", frozenset(["c1"]), e2.where, 0.1, 0.5, 1, "made")
    assert not S._same_place(a, b)
    assert S._same_place(a._replace(periods=PERIODS), b._replace(periods=PERIODS))
    shut = [(1, e2.where, 0)]
    solid = Event("solid", 0.01, 0.1, e1.where, np.array([0]), 1, "body")
    assert not S._reopening(solid, shut, 0.1, 1)
    assert S._reopening(solid, shut, 0.1, 1, PERIODS)


def _drop(periods, angle=60.0):
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


def test_the_normal_part_of_a_displacement_across_the_corner():
    rng = np.random.default_rng(3)
    out, dx = [], None
    for periods in (None, PERIODS):
        ev = _drop(periods)
        if dx is None:
            dx = rng.normal(size=ev.vertices.shape)
        out.append(recipes._normal_part(ev, dx))
    assert np.allclose(out[0], out[1], atol=1e-13)


def test_moves_to_other_images_are_no_displacement():
    from pysurfaceevolver import _periodic
    ev = _drop(PERIODS)
    before = S._places(ev)
    m = ev.mesh()
    _periodic.move_near(ev, [int(m.vertex_ids[3]), int(m.vertex_ids[9])],
                        m.vertices[[3, 9]] + [[PERIODS[0], 0, 0], [0, -PERIODS[1], 0]])
    assert np.abs(S._moved_since(ev, before)).max() < 1e-13


def _bead(periods):
    """test_periodic's bridge from a floor to a bead, on the cell's corner."""
    R, c = 0.5, np.array([0.0, 0.0, 0.55])
    zs = np.linspace(0, 0.35, 6)
    sh = shapes.revolve(np.linspace(0.5, np.sqrt(R*R - 0.2**2), 6), zs, nt=24)
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
    ev.body(1).target = ev.body(1).volume
    return ev


def test_vertices_by_a_sphere_are_moved_home_before_they_are_guarded():
    found = []
    for periods in (None, PERIODS):
        ev = _bead(periods)
        energy = ev.energy
        m = ev.mesh()
        S._home_by_spheres(ev, ["bead"])
        ev.command("recalc")
        m2 = ev.mesh()
        assert abs(ev.energy - energy) < 1e-12 and m2.edge_lengths().max() < 0.2
        near = np.linalg.norm(m2.delta([0, 0, 0.55], m2.vertices), axis=1) < 0.9
        assert np.allclose(m2.vertices[near] - [0, 0, 0.55],
                           m2.delta([0, 0, 0.55], m2.vertices[near]))
        assert np.allclose(m2.delta(m.vertices, m2.vertices), 0.0)
        S.guard(ev)
        _, _, dist, names = ev._constraint_gaps(guards=True)
        guarded = [i for i, n in enumerate(names) if str(n).startswith("pyse_guard")]
        found.append(int(np.isnan(dist[:, guarded]).sum()))
        S._unguard(ev)
    assert found[0] > 0 and found[0] == found[1]


def test_a_vertex_put_on_a_sphere_is_moved_home_first():
    ev = _bead(PERIODS)
    m = ev.mesh()
    free = np.flatnonzero(~np.isnan(ev._constraint_gaps()[2]).any(axis=1))
    far = free[np.abs(m.vertices[free, 0]) > 1.0]     # (wrapped across x = 0)
    assert len(far)
    assert S._move_home(ev, m, far, "bead")
    x = ev.mesh().vertices[far]
    assert np.abs(x[:, :2]).max() < 1.0
    assert not S._move_home(ev, ev.mesh(), far, "bead")
    assert not S._move_home(ev, ev.mesh(), far, "bottom")


def _drop_under_a_sphere(periods):
    drop = shapes.cap(0.7, 0.7, nr=6, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(drop.vertices, periods), drop.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                                2: C.sphere((0, 0, 2.0), 1.0, contact_angle=90.0)},
                   vertex_constraints={1: drop.rims["rim"]},
                   bodies=[pyse.Body(volume=None)], periods=periods)
    ev.body(1).target = ev.body(1).volume
    return ev


def test_a_dome_over_a_footprint_across_the_corner():
    # (_domed's ray cast: the facets at their images by the strip's middle,
    # the sphere from its centre)
    drop = shapes.cap(0.7, 0.7, nr=6, nt=24)
    rim = np.asarray(drop.rims["rim"])
    plain = np.asarray(drop.vertices)
    for a, b in ((rim[np.argmax(plain[rim, 0])], rim[np.argmin(plain[rim, 0])]),
                 (rim[np.argmax(plain[rim, 1])], rim[np.argmax(plain[rim, 0])])):
        for periods in (None, WIDE):
            ev = _drop_under_a_sphere(periods)
            m = ev.mesh()
            nums, _, _, names = ev._constraint_gaps()
            h = float(np.median(m.edge_lengths()))
            assert S._domed(ev, m, np.asarray(m.vertices), int(a), int(b), int(nums[0]), 1, h)


def test_the_floor_with_a_bead_isnt_sunk_across_the_corner():
    for periods in (None, PERIODS):
        assert not _bead(periods)._sunk()


def _edits_refused(monkeypatch):
    """The checked edits (C) refused, both ways alike: Evolver's own commands
    make the changes instead (dissolving, facet_merge)."""
    from pysurfaceevolver import _edits
    monkeypatch.setattr(_edits, "edit", lambda ev, kind, ids, **kw: np.full(max(len(ids), 1), 1))


def _in_cell(e, periods):
    """The event e (found without periods) at its place in the cell."""
    where = np.asarray(e.where, float)
    if periods is not None:
        where = _into_cell(where[None], periods)[0]
    return Event(e.kind, e.gap, e.scale, where, e.vertices, e.constraint, e.between)


def _drop_by_a_sphere(periods):
    """A drop on the floor, on the cell's corner, its top 0.04 under a sphere
    (at 60 degrees)."""
    drop = shapes.cap(0.7, 0.96, nr=6, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(drop.vertices, periods), drop.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                                2: C.sphere((0.0, 0.0, 2.0), 1.0, contact_angle=60.0)},
                   vertex_constraints={1: drop.rims["rim"]},
                   bodies=[pyse.Body(volume=None)], periods=periods)
    ev.body(1).target = ev.body(1).volume
    return ev


def test_a_drop_wets_a_sphere_across_the_corner_as_without_periods(monkeypatch):
    # (the vertices put on the sphere moved home first: its integrals are
    # evaluated there)
    _edits_refused(monkeypatch)
    (e,) = [e for e in _drop_by_a_sphere(None).events() if e.kind == "solid"]
    found = []
    for periods in (None, WIDE):
        ev = _drop_by_a_sphere(periods)
        S.contact(ev, _in_cell(e, periods))
        ev.command("recalc")
        m = ev.mesh()
        on = ev.on_constraint(2)
        assert on.sum() > 10 and np.abs(m.vertices[on, :2]).max() < 0.5
        assert m.edge_lengths().max() < 0.4
        found.append((ev.counts["facets"], ev.energy, ev.body(1).volume, ev.body(1).volconst,
                      m.facet_areas().sum()))
        ev.command("g 1")
        found.append((ev.energy, ev.body(1).volume))
    assert found[0][0] == found[2][0]
    assert np.allclose(found[0][1:], found[2][1:], rtol=1e-12)
    assert np.allclose(found[1], found[3], rtol=1e-12)


def test_balls_merge_across_the_side_as_without_periods(monkeypatch):
    _edits_refused(monkeypatch)
    found = []
    for periods in (None, PERIODS):
        verts, f, nf = _two_balls(periods)
        ev = pyse.Evolver()
        ev.load_arrays(_into_cell(verts, periods), f,
                       bodies=[pyse.Body(faces=range(nf), volume=None),
                               pyse.Body(faces=range(nf, 2*nf), volume=None)], periods=periods)
        for i in (1, 2):
            ev.body(i).target = ev.body(i).volume
        total = ev.bodies().volume.sum()
        touch = Event("touch", 0.02, 0.15, np.array([0.0, 1.0, 1.0]), np.array([0]))
        body = S.merge(ev, _in_cell(touch, periods))
        assert list(ev.bodies().ids) == [body] == [1]
        assert ev.body(1).target == pytest.approx(total, rel=1e-12)
        ev.command("g 10")
        found.append((ev.counts["facets"], ev.energy, ev.body(1).volume,
                      ev.mesh().facet_areas().sum()))
    assert found[0][0] == found[1][0]
    assert np.allclose(found[0][1:], found[1][1:], rtol=1e-12)


# ---- the changes, end to end (the C edits, events and checks in slabs) ----------


def _dumbbell(periods, neck=0.06):
    """A closed body, two lobes joined by a thin neck on the cell's corner."""
    t = np.linspace(0, np.pi, 25)
    z = -1.0*np.cos(t)
    r = 0.5*np.sin(t)*(neck + (1 - neck)*np.cos(t)**2)
    sh = shapes.revolve(np.where(r < 1e-12, 0.0, r), z, 16)
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(sh.vertices, periods), sh.faces, bodies=[pyse.Body(volume=None)],
                   periods=periods)
    ev.body(1).target = ev.body(1).volume
    return ev


def test_a_neck_across_the_corner_pinches_off_as_without_periods():
    found = []
    for periods in (None, WIDE):
        ev = _dumbbell(periods)
        necks = [e for e in ev.events() if e.kind == "neck"]
        assert necks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            made = ev.pinch_off(necks[0])
        assert len(made) == 1
        found.append(sorted(ev.bodies().volume))
    # (the symmetric neck's closest pairs tie: round-off picks the place, and
    # the cut, differently; the liquid is the same)
    assert sum(found[0]) == pytest.approx(sum(found[1]), rel=1e-10)
    assert np.allclose(found[0], found[1], rtol=1e-3)


def test_a_neck_collapsed_below_the_mesh_across_the_corner_is_cut():
    found = []
    for periods in (None, HOUR):
        ev = _hourglass(1e-5, periods)
        volume = ev.body(1).volume
        assert S._pinch_collapsed(ev) == 1
        bodies = list(ev.bodies().ids)
        assert len(bodies) == 2
        assert sum(ev.body(b).target for b in bodies) == pytest.approx(volume, rel=1e-6)
        found.append(sorted(ev.body(b).target for b in bodies))
    assert np.allclose(found[0], found[1], rtol=1e-9)


def _two_drops(periods, theta, gap, R=0.84):
    """test_topology's two drops on a plane, their gap on the cell's side x = 0."""
    t = np.radians(theta)
    drop = shapes.cap(R, R*(1 - np.cos(t)), nr=6, nt=24)
    n, nf = len(drop.vertices), len(drop.faces)
    rim = np.asarray(drop.rims["rim"])
    s = (R*np.sin(t) if theta <= 90 else R) + gap/2
    v = np.r_[drop.vertices - [s, 0, 0], drop.vertices + [s, 0, 0]] + [0, 1.5, 0]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), np.r_[drop.faces, drop.faces + n],
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=theta)},
                   vertex_constraints={1: np.r_[rim, rim + n]},
                   bodies=[pyse.Body(faces=range(nf), volume=None),
                           pyse.Body(faces=range(nf, 2*nf), volume=None)], periods=periods)
    for i in (1, 2):
        ev.body(i).target = ev.body(i).volume
    return ev


def test_drops_merging_across_the_side_are_one_cap_as_without_periods():
    found = []
    for periods in (None, (5.0, 3.0)):
        ev = _two_drops(periods, 120.0, 0.3)
        other = ev.body(2).volume
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev.relax()
            steps = list(recipes.arclength(ev, ev.body(1), 0.2, 12, topology=True))
        fell = [s for s in steps if s.fell]
        assert len(fell) == 1 and fell[0].changes[0].kind == "touch"
        k = steps.index(fell[0])
        assert fell[0].volumes == pytest.approx([steps[k - 1].value + other], rel=1e-8)
        found.append((steps[-1].value, steps[-1].energy, steps[-1].pressures[0]))
    assert np.allclose(found[0], found[1], rtol=1e-3)


def test_a_drop_growing_onto_a_sphere_across_the_corner_wets_it():
    # (test_topology's drop under a sphere: the bridge it becomes, at volume
    # 2.5, the exact one: pressure 1.396298, energy 5.812735)
    for periods in (None, WIDE):
        drop = shapes.cap(0.7, 0.7, nr=6, nt=24)
        ev = pyse.Evolver()
        ev.load_arrays(_into_cell(drop.vertices, periods), drop.faces,
                       constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                                    2: C.sphere((0, 0, 2.0), 1.0, contact_angle=60.0)},
                       vertex_constraints={1: drop.rims["rim"]},
                       bodies=[pyse.Body(volume=None)], periods=periods)
        ev.body(1).target = ev.body(1).volume
        ev.refine(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev.relax()
            for step in recipes.arclength(ev, ev.body(1), 0.1, 20, topology=True):
                if step.fell:
                    break
            assert step.fell and [e.kind for e in step.changes] == ["solid"]
            assert step.changes[0].between == "outside" and step.changes[0].constraint == 2
            ev.body(1).target = 2.5
            assert ev.relax().converged
        assert ev.body(1).pressure == pytest.approx(1.396298, rel=0.003)
        assert ev.energy == pytest.approx(5.812735, rel=0.005)


def _layer(periods, h=0.4, n=10):
    """A liquid layer over the whole floor: its free surface a periodic sheet
    (no contact line), wavy at the start."""
    Lx, Ly = periods
    nx, ny = n, int(round(n*Ly/Lx))
    gx, gy = np.meshgrid(np.arange(nx)*Lx/nx, np.arange(ny)*Ly/ny, indexing="ij")
    z = h + 0.05*np.sin(2*np.pi*gx/Lx)*np.cos(2*np.pi*gy/Ly)
    v = np.c_[gx.ravel(), gy.ravel(), z.ravel()]
    f = []
    for i in range(nx):
        for j in range(ny):
            a, b = i*ny + j, ((i + 1) % nx)*ny + j
            c, d = ((i + 1) % nx)*ny + (j + 1) % ny, i*ny + (j + 1) % ny
            f += [[a, b, c], [a, c, d]]
    ev = pyse.Evolver()
    ev.load_arrays(v, np.array(f), bodies=[pyse.Body(volume=None)], periods=periods)
    ev.body(1).target = h*Lx*Ly
    return ev


def _has_mumps():
    current = pyse.solver()
    try:
        pyse.set_solver("mumps")
        return True
    except ValueError:
        return False
    finally:
        pyse.set_solver(current)


# (Evolver's own factoring doesn't take a flat sheet's Newton steps to an
# equilibrium: its vertices slide freely along it, and the run stops at its
# first point; MUMPS does)
@pytest.mark.skipif(not _has_mumps(), reason="built without MUMPS")
@pytest.mark.parametrize("events", [False, True])
def test_arclength_on_a_periodic_liquid_layer(events):
    # (a flat layer at any volume: its area the cell's, no pressure)
    ev = _layer(PERIODS)
    area = PERIODS[0]*PERIODS[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ev.relax().converged
        steps = list(recipes.arclength(ev, ev.body(1), 0.3, 5, events=events))
    # (slowly at first: a flat sheet's vertices slide freely along it, and
    # Newton's first steps make little way)
    assert len(steps) == 5
    assert all(b.value > a.value for a, b in zip(steps, steps[1:]))
    for s in steps:
        assert s.volumes[0] == pytest.approx(s.value, rel=1e-8)
        assert s.energy == pytest.approx(area, rel=1e-8)
        assert abs(s.pressures[0]) < 1e-6
    z = ev.vertices[:, 2]
    assert np.ptp(z) < 1e-6 and z.mean() == pytest.approx(steps[-1].value/area, rel=1e-8)
