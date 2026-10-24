"""Events (Evolver.events, _events.py and pyse_events.c) and the mesh
geometry of _geometry.py in slabs periodic in x and y: the same as without
periods, features straddling the cell's sides and corner, and a surface
coming close to its own image."""

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import _core, _periodic
from pysurfaceevolver import constraints as C
from pysurfaceevolver import shapes
from pysurfaceevolver._geometry import _edge_turn, _self_gap


def _into_cell(v, periods):
    if periods is None:
        return v
    L = np.asarray(periods)
    v = v.copy()
    v[:, :2] -= L*np.floor(v[:, :2]/L)
    return v


def _same(plain, periodic, periods, rows=None):
    """The same events: kinds, gaps, scales, constraints, what lies between,
    the vertices (rows mapped by rows) and the places (at any image)."""
    assert [e.kind for e in plain] == [e.kind for e in periodic]
    for p, q in zip(plain, periodic):
        assert q.gap == pytest.approx(p.gap, abs=1e-12)
        assert q.scale == pytest.approx(p.scale, abs=1e-12)
        assert (p.constraint, p.between) == (q.constraint, q.between)
        assert np.abs(_periodic.delta(p.where, q.where, periods)).max() < 1e-12
        mapped = p.vertices if rows is None else np.asarray(rows)[p.vertices]
        assert sorted(set(mapped.tolist())) == sorted(q.vertices.tolist())


def _both(make, periods):
    return make(None).events(), make(periods).events()


def _ball(r=0.5, nt=16):
    t = np.linspace(0, np.pi, 9)
    return shapes.revolve(r*np.sin(t), -r*np.cos(t), nt)


def _two_balls(periods, gap=0.02):
    """Two balls gap apart across the cell's corner."""
    ball = _ball()
    n, nf = len(ball.vertices), len(ball.faces)
    s = [0.5 + gap/2, 0, 0]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(np.r_[ball.vertices - s, ball.vertices + s], periods),
                   np.r_[ball.faces, ball.faces + n],
                   bodies=[pyse.Body(faces=range(nf), volume=None),
                           pyse.Body(faces=range(nf, 2*nf), volume=None)], periods=periods)
    return ev


def _hourglass(periods):
    """A body with a narrow waist on the cell's corner (its axis on z; the
    vertices jiggled, so that one pair is the closest, not a ring's to
    round-off)."""
    z = np.linspace(-1.2, 1.2, 25)
    r = (0.5 - 0.47*np.exp(-(z/0.3)**2))*np.sqrt(np.clip(1 - (z/1.2)**8, 0, None))
    r[0] = r[-1] = 0
    sh = shapes.revolve(r, z, 24)
    v = sh.vertices + 1e-3*np.random.default_rng(1).uniform(-1, 1, sh.vertices.shape)
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), sh.faces, bodies=[pyse.Body(volume=None)],
                   periods=periods)
    return ev


def _two_drops(periods, theta=60.0, gap=0.02, R=0.4):
    """Two drops on the floor, their contact lines (theta <= 90) or their
    equators gap apart across x = 0, on the cell's corner."""
    t = np.radians(theta)
    drop = shapes.cap(R, R*(1 - np.cos(t)), nr=6, nt=24)
    n, nf = len(drop.vertices), len(drop.faces)
    rim = np.asarray(drop.rims["rim"])
    s = [(R*np.sin(t) if theta <= 90 else R) + gap/2, 0, 0]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(np.r_[drop.vertices - s, drop.vertices + s], periods),
                   np.r_[drop.faces, drop.faces + n],
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=theta)},
                   vertex_constraints={1: np.r_[rim, rim + n]},
                   bodies=[pyse.Body(faces=range(nf), volume=None),
                           pyse.Body(faces=range(nf, 2*nf), volume=None)], periods=periods)
    return ev


def _floating(periods):
    """A ball just above the floor, on the cell's corner."""
    ball = _ball()
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(ball.vertices + [0, 0, 0.52], periods), ball.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=90.0)},
                   bodies=[pyse.Body(volume=None)], periods=periods)
    return ev


def test_balls_about_to_touch_across_the_corner():
    plain, periodic = _both(_two_balls, (3.0, 2.0))
    assert [e.kind for e in plain] == ["touch"] and plain[0].gap == pytest.approx(0.02)
    _same(plain, periodic, (3.0, 2.0))


def test_a_neck_on_the_cells_corner():
    plain, periodic = _both(_hourglass, (1.6, 1.4))
    assert [e.kind for e in plain] == ["neck"]
    _same(plain, periodic, (1.6, 1.4))
    # its waist's vertices in all four quadrants, wrapped into the cell
    x = _hourglass((1.6, 1.4)).vertices[periodic[0].vertices]
    assert (x[:, 0] > 0.8).any() and (x[:, 0] < 0.8).any()
    assert (x[:, 1] > 0.7).any() and (x[:, 1] < 0.7).any()


@pytest.mark.parametrize("theta", [60.0, 120.0])
def test_drops_meeting_across_the_cells_side(theta):
    # (60: their contact lines close; 120: their equators, touching across
    # the outside)
    plain, periodic = _both(lambda p: _two_drops(p, theta), (2.0, 1.2))
    assert [e.kind for e in plain] == (["contact lines"] if theta < 90 else ["touch"])
    _same(plain, periodic, (2.0, 1.2))


def test_drops_meeting_after_steps():
    # the same steps either way, then the same events
    found = []
    for periods in (None, (2.0, 1.2)):
        ev = _two_drops(periods, 60.0, 0.05)
        for k in (1, 2):
            ev.body(k).target = ev.body(k).volume
        ev.command("g 5")
        found.append(ev.events(1.0))
    assert found[0]
    _same(found[0], found[1], (2.0, 1.2))


def test_a_ball_near_the_floor_on_the_corner():
    plain, periodic = _both(_floating, (2.0, 1.6))
    assert [e.kind for e in plain] == ["solid"] and plain[0].gap == pytest.approx(0.02)
    _same(plain, periodic, (2.0, 1.6))


def _drop_and_its_image(periods, theta, gap=0.02, R=0.4):
    """A drop whose cell is just wider than it (its contact line or its
    equator gap from its image's), or without periods, the drop and its
    image."""
    t = np.radians(theta)
    drop = shapes.cap(R, R*(1 - np.cos(t)), nr=6, nt=24)
    Lx = 2*(R*np.sin(t) if theta <= 90 else R) + gap
    v, f = drop.vertices + [0.3, 0.2, 0], drop.faces
    rim = np.asarray(drop.rims["rim"])
    cons = {1: C.plane((0, 0, 1), 0.0, contact_angle=theta)}
    ev = pyse.Evolver()
    if periods is None:
        n, nf = len(v), len(f)
        ev.load_arrays(np.r_[v, v + [Lx, 0, 0]], np.r_[f, f + n], constraints=cons,
                       vertex_constraints={1: np.r_[rim, rim + n]},
                       bodies=[pyse.Body(faces=range(nf), volume=None),
                               pyse.Body(faces=range(nf, 2*nf), volume=None)])
    else:
        ev.load_arrays(_into_cell(v, (Lx, 1.5)), f, constraints=cons, vertex_constraints={1: rim},
                       bodies=[pyse.Body(volume=None)], periods=(Lx, 1.5))
    return ev, (Lx, 1.5)


@pytest.mark.parametrize("theta", [60.0, 120.0])
def test_a_drop_close_to_its_own_image(theta):
    # the slab's drop is the plain drop and its image: the same events, its
    # vertices once
    plain = _drop_and_its_image(None, theta)[0].events()
    ev, periods = _drop_and_its_image(True, theta)
    periodic = ev.events()
    assert plain and plain[0].kind == ("contact lines" if theta < 90 else "touch")
    assert plain[0].gap < 0.04
    n = len(ev.vertices)
    _same(plain, periodic, periods, rows=np.r_[np.arange(n), np.arange(n)])


def _ridge(periods, w, shift=0.0, ns=12, ny=12):
    """A liquid ridge along y across the cell (a periodic layer), contact
    angle 90: a half cylinder of width w on the floor, its axis at x =
    shift."""
    Ly = periods[1]
    phi = np.linspace(np.pi, 0, ns + 1)
    xs, zs = shift + w/2*np.cos(phi), w/2*np.sin(phi)
    ys = Ly*np.arange(ny)/ny
    v = np.array([[x, y, z] for y in ys for x, z in zip(xs, zs)])
    f = []
    for j in range(ny):
        for i in range(ns):
            a, b = j*(ns + 1) + i, ((j + 1) % ny)*(ns + 1) + i
            f += [[a, a + 1, b + 1], [a, b + 1, b]]
    f = np.array(f)
    c = v[f]
    d = _periodic.delta
    n = np.cross(d(c[:, 0], c[:, 1], periods), d(c[:, 0], c[:, 2], periods))
    if (n[:, 2] < 0).sum() > len(n)/2:
        f = f[:, ::-1]
    rim = [j*(ns + 1) for j in range(ny)] + [j*(ns + 1) + ns for j in range(ny)]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), f,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=90.0)},
                   vertex_constraints={1: rim}, bodies=[pyse.Body(volume=None)], periods=periods)
    return ev


def test_a_periodic_layer_whose_sides_come_close_across_a_period():
    # a ridge across the cell: its two sides, a period apart, close at the
    # floor (its contact lines) and above it (touching across the outside)
    periods = (1.0, 0.9)
    for shift in (0.0, 0.5, 0.31):
        found = _ridge(periods, 0.97, shift).events()
        assert sorted(e.kind for e in found) == ["contact lines", "touch"]
        lines = [e for e in found if e.kind == "contact lines"][0]
        assert lines.gap == pytest.approx(0.03, abs=1e-12) and lines.constraint == 1
        assert abs(_periodic.delta(lines.where, [shift + 0.5, 0, 0], periods)[0]) < 1e-12
        if shift == 0.0:
            first = found
        else:
            by_kind = [sorted(evs, key=lambda e: e.kind) for evs in (first, found)]
            for p, q in zip(*by_kind):
                assert q.gap == pytest.approx(p.gap, abs=1e-12)
                d = _periodic.delta(p.where, q.where, periods)
                assert abs(d[0] - shift) < 1e-12 or abs(abs(d[0] - shift) - 1.0) < 1e-12
    assert not _ridge(periods, 0.6).events()


def _nearest_segment(p, a, b, periods):
    """Each point's distance to the nearest of the segments' images, by brute
    force (the images within a few periods)."""
    best = np.full(len(p), np.inf)
    for i in range(-5, 6):
        for j in range(-7, 8):
            s = np.array([i*periods[0], j*periods[1], 0.0])
            for k in range(len(a)):
                ab = b[k] - a[k]
                q = p - (a[k] + s)
                t = np.clip(q @ ab/(ab @ ab), 0, 1)
                best = np.minimum(best, np.linalg.norm(q - t[:, None]*ab, axis=1))
    return best


def test_distances_to_segments_at_their_images():
    rng = np.random.default_rng(3)
    periods = (1.3, 0.9)
    a = rng.uniform(-2, 3, (40, 3))
    b = a + rng.uniform(-0.3, 0.3, (40, 3))
    p = rng.uniform(-2, 3, (200, 3))
    for n in (3, 40):           # (each point against all of them; the tree)
        got = _core.events_segments(p, a[:n], b[:n], periods)
        assert np.abs(got - _nearest_segment(p, a[:n], b[:n], periods)).max() < 1e-12


def test_the_self_gap_and_edge_turns_across_the_sides():
    periods = (2.0, 1.2)
    plain, periodic = _two_drops(None, 120.0, 0.03).mesh(), _two_drops(periods, 120.0, 0.03).mesh()
    assert np.array_equal(plain.edges, periodic.edges)
    turn0, judged0 = _edge_turn(plain)
    turn1, judged1 = _edge_turn(periodic)
    assert np.array_equal(judged0, judged1) and np.abs(turn0 - turn1).max() < 1e-9
    used = np.ones(len(plain.vertices), dtype=bool)
    h = float(plain.edge_lengths().mean())
    g0 = _self_gap(plain.vertices, plain.edges, used, h)
    g1 = _self_gap(periodic.vertices, periodic.edges, used, h, periods)
    assert g0 < 0.5 and g1 == pytest.approx(g0, abs=1e-12)
    # (without the periods the wrapped vertices' gap is across the cell)
    assert _self_gap(periodic.vertices, periodic.edges, used, h) > g0
