"""The liquid's clusters (pysurfaceevolver._clusters): a body's surface
pieces joined through the liquid between them, and the bodies a topology
run makes of them."""

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import _clusters as K, constraints as C, shapes

_WALLS = {"xlo": ((1, 0, 0), 0.0), "xhi": ((-1, 0, 0), -1.0),
          "ylo": ((0, 1, 0), 0.0), "yhi": ((0, -1, 0), -1.0)}


def _square(z, up, n=10):
    """The square [0, 1]^2 at height z in n x n cells of two triangles,
    facing up (or down); and the rows on each side wall."""
    t = np.linspace(0, 1, n + 1)
    x, y = np.meshgrid(t, t, indexing="ij")
    v = np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]
    k = np.arange((n + 1)*(n + 1)).reshape(n + 1, n + 1)
    a, b, c, d = k[:-1, :-1].ravel(), k[1:, :-1].ravel(), k[1:, 1:].ravel(), k[:-1, 1:].ravel()
    f = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    if not up:
        f = f[:, [0, 2, 1]]
    tol = 1e-9
    sides = {"xlo": np.flatnonzero(v[:, 0] < tol), "xhi": np.flatnonzero(v[:, 0] > 1 - tol),
             "ylo": np.flatnonzero(v[:, 1] < tol), "yhi": np.flatnonzero(v[:, 1] > 1 - tol)}
    return v, f, sides


def _ball(centre, r):
    t = np.linspace(0, np.pi, 9)
    ball = shapes.revolve(r*np.sin(t), -r*np.cos(t), 16)
    return np.asarray(ball.vertices, float) + centre, np.asarray(ball.faces)


def _load(parts, floor=False, sides=None):
    """One body of the pieces (vertices, faces), side walls at 90 degrees
    where ``sides`` (the rows on each, by piece), the floor z = 0 a wall
    "bottom" with ``floor``."""
    v, f, on = [], [], {}
    n = 0
    for i, (pv, pf) in enumerate(parts):
        v.append(pv)
        f.append(pf + n)
        for key, rows in (sides[i] if sides and sides[i] else {}).items():
            on.setdefault(key, []).extend((np.asarray(rows) + n).tolist())
        n += len(pv)
    cons = {k: C.plane(nrm, d, contact_angle=90.0) for k, (nrm, d) in _WALLS.items()} \
        if sides else {}
    if floor:
        cons["bottom"] = C.plane((0, 0, 1), 0.0, contact_angle=90.0)
    ev = pyse.Evolver()
    ev.load_arrays(np.concatenate(v), np.concatenate(f), constraints=cons,
                   vertex_constraints=on, bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    return ev


def test_components_label_each_node_by_its_smallest():
    assert K.components(6, [0, 2, 3], [5, 3, 4]).tolist() == [0, 1, 2, 2, 2, 0]
    assert K.components(3, [], []).tolist() == [0, 1, 2]


def test_two_drops_in_one_body_are_two_clusters():
    ev = _load([_ball((0, 0, 0), 1.0), _ball((3, 0, 0), 0.7)])
    c = K.clusters(ev)
    assert len(c.bodies) == 2 and c.bodies == [frozenset({1})]*2
    assert c.mixed == 0
    # each drop's grid nodes about its volume
    sizes = sorted(c.nodes*c.spacing**3)
    assert sizes == pytest.approx([4/3*np.pi*0.7**3, 4/3*np.pi], rel=0.25)


def test_a_bubble_inside_the_liquid_is_no_cluster_of_its_own():
    outer = _ball((0, 0, 0), 1.0)
    inner = _ball((0, 0, 0), 0.5)
    ev = _load([outer, (inner[0], inner[1][:, [0, 2, 1]])])
    assert ev.body(1).volume == pytest.approx(4/3*np.pi*(1 - 0.125), rel=0.1)
    c = K.clusters(ev)
    assert len(c.bodies) == 1 and c.mixed == 0


def test_two_menisci_of_one_throat_are_one_cluster():
    # liquid between walls from z = 0.2 to 0.8: two surface pieces (a
    # meniscus at either end), one liquid; and reaching the floor through
    # the liquid where it fills down to it, no vertex on it
    lo, hi = _square(0.2, up=False), _square(0.8, up=True)
    ev = _load([lo[:2], hi[:2]], floor=True, sides=[lo[2], hi[2]])
    assert ev.body(1).volume == pytest.approx(0.6)
    c = K.clusters(ev, "bottom")
    assert len(c.bodies) == 1 and not c.outlet[0]
    assert c.walls["xlo"][0] > 0 and "bottom" not in c.walls
    top = _square(0.8, up=True)
    ev = _load([top[:2]], floor=True, sides=[top[2]])
    c = K.clusters(ev, "bottom")
    assert len(c.bodies) == 1 and c.outlet[0]


def _run(ev, keep=1, outlet=()):
    ev._liquid = dict(keep=keep, outlet=list(outlet), original=frozenset({1}), dry=False)


def test_reconcile_gives_each_cluster_a_body_and_the_rest_to_the_largest():
    ev = _load([_ball((0, 0, 0), 1.0), _ball((3, 0, 0), 0.7)])
    small = ev.body(1).volume*0.7**3/(1 + 0.7**3)
    ev.body(1).target = ev.body(1).volume + 0.1        # (a step the run has yet to make)
    _run(ev)
    K.reconcile(ev)
    bodies = ev.bodies()
    assert list(bodies.ids) == [1, 2]
    assert bodies.volume[1] == pytest.approx(small)
    assert ev.body(2).target == pytest.approx(bodies.volume[1], rel=1e-12)
    assert ev.body(1).target == pytest.approx(bodies.volume[0] + 0.1, rel=1e-12)
    assert sum(bodies.target) == pytest.approx(sum(bodies.volume) + 0.1, rel=1e-12)
    K.reconcile(ev)                                     # (nothing to do again)
    assert list(ev.bodies().ids) == [1, 2]
    ev._liquid = None


def test_reconcile_keeps_the_liquid_reaching_the_outlet_draining():
    # a thin layer on the floor and a larger drop above it, one body: the
    # layer drains (body 1), the drop is trapped at what it holds
    layer = _square(0.2, up=True)
    ev = _load([layer[:2], _ball((0.5, 0.5, 0.75), 0.45)], floor=True, sides=[layer[2], None])
    total = ev.body(1).volume
    ev.body(1).target = total - 0.01
    _run(ev, outlet=["bottom"])
    K.reconcile(ev)
    assert not ev._liquid["dry"]
    bodies = ev.bodies()
    assert list(bodies.ids) == [1, 2]
    assert bodies.volume[0] == pytest.approx(0.2)
    assert ev.body(1).target == pytest.approx(0.2 - 0.01, rel=1e-9)
    assert ev.body(2).target == pytest.approx(total - 0.2, rel=1e-9)
    # without the outlet the drop is the largest: it would drain
    ev = _load([layer[:2], _ball((0.5, 0.5, 0.75), 0.45)], floor=True, sides=[layer[2], None])
    _run(ev)
    K.reconcile(ev)
    assert ev.body(1).volume == pytest.approx(total - 0.2)
    # and nothing reaching it: the run told
    ev = _load([_ball((0.5, 0.5, 0.75), 0.45)], floor=True)
    _run(ev, outlet=["bottom"])
    K.reconcile(ev)
    assert ev._liquid["dry"] and list(ev.bodies().ids) == [1]
    ev._liquid = None


def _flat(L, z, h):
    """The square [0, L]^2 at height z facing up, in nearly equilateral
    triangles of edge about h (rows offset by half an edge), and the rows
    on each side."""
    dy = h*np.sqrt(3)/2
    ny = max(2, int(round(L/dy)))
    nx = max(2, int(round(L/h)))
    rows, v = [], []
    for j, y in enumerate(np.linspace(0, L, ny + 1)):
        xs = np.linspace(0, L, nx + 1)
        if j % 2 and 0 < j < ny:
            xs = np.r_[0.0, (xs[:-1] + xs[1:])/2, L]
        rows.append(list(range(len(v), len(v) + len(xs))))
        v += [(x, y, z) for x in xs]
    v = np.array(v)
    f = []
    for a, b in zip(rows[:-1], rows[1:]):
        i = k = 0
        while i < len(a) - 1 or k < len(b) - 1:
            if k == len(b) - 1 or (i < len(a) - 1 and v[a[i + 1], 0] <= v[b[k + 1], 0]):
                f.append((a[i], a[i + 1], b[k]))
                i += 1
            else:
                f.append((a[i], b[k + 1], b[k]))
                k += 1
    f = np.array(f)
    tol = 1e-9
    sides = {"xlo": np.flatnonzero(v[:, 0] < tol), "xhi": np.flatnonzero(v[:, 0] > L - tol),
             "ylo": np.flatnonzero(v[:, 1] < tol), "yhi": np.flatnonzero(v[:, 1] > L - tol)}
    return v, f, sides


def test_a_pendular_ring_cut_off_from_the_draining_liquid_keeps_its_volume():
    # two beads stacked in a box, the liquid over them drained through the
    # floor: the meniscus goes down past the upper bead and leaves a ring
    # where the beads touch, cut off; it keeps its volume while the liquid
    # round the lower bead drains on to the floor, and relaxed alone it is
    # the same (its own pressure, not the draining liquid's)
    import warnings
    from pysurfaceevolver import recipes
    L, theta, h = 3.0, 20.0, 0.3
    beads = [(L/2, L/2, 1.015), (L/2, L/2, 3.045)]
    v, f, sides = _flat(L, 4.05, h)
    walls = {"xlo": C.plane((1, 0, 0), 0.0, contact_angle=90.0),
             "xhi": C.plane((-1, 0, 0), -L, contact_angle=90.0),
             "ylo": C.plane((0, 1, 0), 0.0, contact_angle=90.0),
             "yhi": C.plane((0, -1, 0), -L, contact_angle=90.0),
             "bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0)}
    for i, c in enumerate(beads):
        walls[f"bead{i + 1}"] = C.sphere(c, 1.0, contact_angle=theta, gauge="flux")
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints=walls, vertex_constraints=sides,
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.relax()
        run = []
        for step in recipes.arclength(ev, ev.body(1), -0.5, 200, topology=True,
                                      outlet="bottom"):
            run.append((step, ev.bodies()))
            if any(e.kind == "solid" and str(e.constraint) == "bottom" and e.between == "body"
                   for e in step.events) or ev.on_constraint("bottom").any():
                break
    cut = [i for i, (s, _) in enumerate(run) if s.trapped > 0]
    assert cut, "no ring cut off"
    after = run[cut[0]:]
    held = after[0][0].trapped
    assert 0.1 < held < 0.5 and len(after) >= 5
    for step, bodies in after:
        assert list(bodies.ids) == [1, 2]
        assert step.trapped == pytest.approx(held, rel=1e-9)
        assert bodies.target[1] == pytest.approx(held, rel=1e-12)
    assert after[-1][0].value < after[0][0].value - 2       # (the floor's liquid drained on)
    ring = ev.mesh()
    on = [k for k in ("bead1", "bead2") if ev.on_constraint(k)[
        np.unique(np.asarray(ring.facets)[(np.asarray(ring.facet_bodies) == 2).any(axis=1)])].any()]
    assert on == ["bead1", "bead2"]
    pressure = ev.body(2).pressure
    gone = np.asarray(ring.facet_ids)[(np.asarray(ring.facet_bodies) == 1).any(axis=1)]
    ev.command("; ".join(f"dissolve facet[{int(i)}]" for i in gone))
    ev.command("dissolve edges; dissolve vertices; dissolve body[1]")
    assert ev.relax().converged
    assert ev.body(2).volume == pytest.approx(held, rel=1e-9)
    assert ev.body(2).pressure == pytest.approx(pressure, rel=1e-5)
