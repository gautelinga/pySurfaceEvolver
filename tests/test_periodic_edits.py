"""The checked mesh edits, the remesher, the crossings and the topology
check in slabs periodic in x and y (bindings/pyse_edit.c, pyse_check.c):
the same surface with and without periods, its feature across the cell's
corner (its vertices wrapped into the cell), gives the same results."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import _core, constraints as C, shapes
from pysurfaceevolver._edits import crossing_places, crossings, edit, remesh, walls
from pysurfaceevolver._surgery import check_topology

warnings.simplefilter("ignore")

PERIODS = (1.9, 2.3)


def _into_cell(v, periods):
    if periods is None:
        return v
    L = np.asarray(periods)
    v = np.array(v, dtype=float)
    v[:, :2] -= L*np.floor(v[:, :2]/L)
    return v


def _drop(periods):
    """A sessile drop on a floor on the cell's corner, its vertices jiggled
    (no two edges alike: the edits' orders by length the same with periods)."""
    cap = shapes.cap(0.5, 0.3, nr=6, nt=24)
    v = cap.vertices + 0.01*np.random.default_rng(0).standard_normal(cap.vertices.shape)
    rim = np.asarray(cap.rims["rim"])
    v[rim, 2] = 0.0
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), cap.faces,
                   constraints={"bottom": C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
                   vertex_constraints={"bottom": rim}, bodies=[pyse.Body(volume=None)],
                   periods=periods)
    ev.body(1).target = ev.body(1).volume
    ev.command("g 5")
    return ev


BEAD = np.array([0.0, 0.0, 0.55])


def _bead(periods):
    """A liquid bridge from a floor to a bead above the cell's corner."""
    R = 0.5
    sh = shapes.revolve(np.linspace(0.5, np.sqrt(R*R - 0.2**2), 6), np.linspace(0, 0.35, 6), nt=24)
    v = sh.vertices + 0.005*np.random.default_rng(0).standard_normal(sh.vertices.shape)
    top, bot = np.asarray(sh.rims["top"]), np.asarray(sh.rims["bottom"])
    v[bot, 2] = 0
    d = v[top] - BEAD
    v[top] = BEAD + R*d/np.linalg.norm(d, axis=1)[:, None]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), sh.faces,
                   constraints={"bottom": C.plane((0, 0, 1), 0.0, contact_angle=50.0),
                                "bead": C.sphere(tuple(BEAD), R, contact_angle=40.0,
                                                 axis=(0, 0, 1))},
                   vertex_constraints={"bottom": bot, "bead": top},
                   bodies=[pyse.Body(volume=None)], periods=periods)
    ev.body(1).target = ev.body(1).volume
    ev.command("g 5")
    return ev


def _volumes(ev):
    # (Evolver's, evaluated: in a slab the volumes ev.bodies() reads, kept
    # with the bodies, stay as they were through a recalc after a facet is
    # dissolved, until the next iteration)
    return np.array([float(ev.command(f"print body[{int(b)}].volume")) for b in ev.bodies().ids])


def _state(ev, result=None):
    m = ev.mesh()
    return dict(result=None if result is None else np.asarray(result), mesh=m, energy=ev.energy,
                volumes=_volumes(ev),
                problems=sorted(p[1].split(" (")[0] for p in check_topology(ev)),
                crossings=crossings(ev), places=crossing_places(ev), periods=ev.periods,
                lengths=ev.values("edge", "length") if m.facets is not None else None)


def _both(make, act):
    """act on the surface made without periods and with them (each done
    before the next is made: switching surfaces renumbers them)."""
    out = []
    for periods in (None, PERIODS):
        ev = make(periods)
        out.append(_state(ev, act(ev)))
    return out


def _same(a, b, tol=1e-12):
    ma, mb = a["mesh"], b["mesh"]
    if a["result"] is not None:
        assert np.array_equal(a["result"], b["result"])
    assert np.array_equal(ma.facets, mb.facets) and np.array_equal(ma.edges, mb.edges)
    assert np.abs(mb.delta(ma.vertices, mb.vertices)).max() < tol
    assert abs(a["energy"] - b["energy"]) < tol
    assert np.abs(a["volumes"] - b["volumes"]).max(initial=0) < tol
    assert a["problems"] == b["problems"] and a["crossings"] == b["crossings"]
    assert np.abs(mb.delta(a["places"], b["places"])).max(initial=0) < tol
    _sound(b)


def _sound(s):
    """A slab's surface as its edits must leave it: every edge's wrap its
    ends' nearest images (Evolver's lengths the minimum-image ones)."""
    if s["periods"] is not None:
        assert np.abs(s["lengths"] - s["mesh"].edge_lengths()).max() < 1e-9


def _by_length(ev, n, longest=False):
    m = ev.mesh()
    lengths = m.edge_lengths()
    return np.asarray(m.edge_ids)[np.argsort(-lengths if longest else lengths, kind="stable")[:n]]


EDITS = {
    "split": lambda ev: edit(ev, "split", _by_length(ev, 60, longest=True)),
    "collapse": lambda ev: edit(ev, "collapse", _by_length(ev, 60)),
    "flip": lambda ev: edit(ev, "flip", _by_length(ev, 30, longest=True), max_turn=60),
    "valences": lambda ev: edit(ev, "flip", np.asarray(ev.mesh().edge_ids)[
        ev.mesh().edge_valence() == 2], mode=2),
    "equiangulate": lambda ev: edit(ev, "flip", ev.mesh().edge_ids, mode=1),
    "smooth": lambda ev: edit(ev, "smooth", ev.mesh().vertex_ids, smoothing=0.8),
    "degenerate": lambda ev: edit(ev, "degenerate", ev.mesh().facet_ids, flat=0.3, max_turn=60),
}


# (the bead's bridge is as near valence 6 as flips make it already)
@pytest.mark.parametrize("make, op", [(make, op) for make in (_drop, _bead) for op in sorted(EDITS)
                                      if (make, op) != (_bead, "valences")],
                         ids=lambda x: x if isinstance(x, str) else x.__name__[1:])
def test_each_edit_across_the_corner_is_the_edit_without_periods(make, op):
    a, b = _both(make, EDITS[op])
    assert (a["result"] == 0).sum() > 0
    _same(a, b)


@pytest.mark.parametrize("make", [_drop, _bead], ids=["drop", "bead"])
@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_a_remesh_across_the_corner_is_the_remesh_without_periods(make, factor):
    # (coarser than the mesh: few splits, whose two halves tie on length and
    # are collapsed in either order by round-off)
    def act(ev):
        h = factor*float(np.median(ev.mesh().edge_lengths()))
        return list(remesh(ev, h).values())
    a, b = _both(make, act)
    assert a["result"].sum() > 100
    _same(a, b, tol=1e-11)


def test_a_remesh_near_a_change_across_the_corner():
    def act(ev):
        m = ev.mesh()
        h = float(np.median(m.edge_lengths()))
        near = np.linalg.norm(m.delta(m.vertices[0], m.vertices)[:, :2], axis=1) < 0.3
        return list(remesh(ev, 1.3*h, region=np.flatnonzero(near)).values())
    a, b = _both(_bead, act)
    assert a["result"].sum() > 20
    _same(a, b, tol=1e-11)


@pytest.mark.parametrize("make", [_drop, _bead], ids=["drop", "bead"])
def test_a_finer_remesh_across_the_corner(make):
    # (splits' halves tie, so the edits part ways: alike, not the same)
    def act(ev):
        h = 0.6*float(np.median(ev.mesh().edge_lengths()))
        done = remesh(ev, h)
        if make is _bead:
            # its vertices kept at the image near its centre
            on = ev.on_constraint("bead")
            assert np.abs(np.linalg.norm(ev.mesh().vertices[on] - BEAD, axis=1) - 0.5).max() < 1e-9
        return [h] + list(done.values())
    a, b = _both(make, act)
    for s in (a, b):
        assert s["problems"] == [] and s["crossings"] == 0
        lengths = s["mesh"].edge_lengths()
        h = s["result"][0]
        assert abs(np.median(lengths)/h - 1) < 0.15 and lengths.max() < 2*h
    _sound(b)
    counts = a["result"][1:], b["result"][1:]
    assert np.all(np.abs(counts[0] - counts[1]) <= 0.15*counts[0] + 5)
    assert abs(a["volumes"][0] - b["volumes"][0]) < 2e-3*a["volumes"][0]


def test_a_vertex_pushed_into_the_bead_across_the_corner():
    def act(ev):
        m = ev.mesh()
        on = ev.on_constraint("bead")
        x = m.vertices.copy()
        to = m.delta(x, BEAD)
        k = int(np.argmin(np.where(on, 9, np.linalg.norm(to, axis=1))))
        x[k] += 0.5*to[k]
        ev.vertices = x
    a, b = _both(_bead, act)
    assert "a vertex inside a solid" in a["problems"]
    _same(a, b)


def _two_drops(periods, gap=0.05):
    """Two drops on the floor, either side of the cell's side x = 0 (their
    contact lines gap apart there), wrapped into the cell."""
    R, t = 0.5, np.radians(60.0)
    cap = shapes.cap(R, R*(1 - np.cos(t)), nr=5, nt=20)
    n, nf = len(cap.vertices), len(cap.faces)
    rim = np.asarray(cap.rims["rim"])
    s = R*np.sin(t) + gap/2
    v = np.r_[cap.vertices - [s, -0.3, 0], cap.vertices + [s, 0.3, 0]]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), np.r_[cap.faces, cap.faces + n],
                   constraints={"bottom": C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
                   vertex_constraints={"bottom": np.r_[rim, rim + n]},
                   bodies=[pyse.Body(faces=range(nf), volume=None),
                           pyse.Body(faces=range(nf, 2*nf), volume=None)], periods=periods)
    return ev


def test_crossings_and_their_places_across_the_corner():
    # two drops pushed into each other across the cell's side
    a, b = _both(lambda p: _two_drops(p, gap=-0.3), lambda ev: None)
    assert a["crossings"] > 0 and len(a["places"]) == a["crossings"]
    assert "an edge through a facet" in a["problems"]
    _same(a, b)


def test_bridge_across_the_cells_side():
    def act(ev):
        m = ev.mesh()
        x, e = m.vertices, np.asarray(m.edges)
        border = np.flatnonzero(m.edge_valence() == 1)
        mid = x[e[border, 0]] + m.delta(x[e[border, 0]], x[e[border, 1]])/2
        mid = m.delta(np.zeros(3), mid)                 # (round the corner)
        left, right = border[mid[:, 0] < 0], border[mid[:, 0] > 0]
        i = left[np.argmax(mid[mid[:, 0] < 0][:, 0])]
        j = right[np.argmin(mid[mid[:, 0] > 0][:, 0])]
        return edit(ev, "bridge", [m.edge_ids[i], m.edge_ids[j]])
    a, b = _both(_two_drops, act)
    assert list(a["result"]) == [0]
    _same(a, b)


def _boxes(periods):
    """Two boxes facing each other across the cell's side x = 0."""
    box = shapes.box(n=3)
    v, f = np.asarray(box.vertices, float), np.asarray(box.faces)
    v = 0.5*np.r_[v - [1.1, 0.5, 0.5], v + [0.1, -0.45, -0.47]] + [0, 0, 1]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), np.r_[f, f + len(box.vertices)],
                   bodies=[pyse.Body(faces=range(len(f))),
                           pyse.Body(faces=range(len(f), 2*len(f)))], periods=periods)
    return ev


def test_merge_across_the_cells_side():
    def act(ev):
        m = ev.mesh()
        c = m.facet_corners().mean(axis=1)
        c = m.delta(np.zeros(3), c)
        first = np.arange(len(c)) < len(c)//2
        a = int(np.argmin(np.where(first, np.linalg.norm(c - [-0.05, 0, 1], axis=1), 9)))
        b = int(np.argmin(np.where(~first, np.linalg.norm(c - c[a], axis=1), 9)))
        return edit(ev, "merge", [m.facet_ids[a], m.facet_ids[b]])
    a, b = _both(_boxes, act)
    assert list(a["result"]) == [0]
    _same(a, b)


def _tetras(periods):
    """Two tetrahedra meeting at a vertex on the cell's corner (a pinch)."""
    tv = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]]) + [0, 0, 1]
    tf = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    v = np.r_[tv, [0, 0, 2] - tv[1:]]
    f = np.r_[tf, np.where(tf == 0, 0, tf + 3)[:, ::-1]]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), f,
                   bodies=[pyse.Body(faces=range(4)), pyse.Body(faces=range(4, 8))],
                   periods=periods)
    return ev


def test_separate_and_snap_on_the_cells_corner():
    def act(ev):
        assert "a pinched vertex" in [p[1].split(" (")[0] for p in check_topology(ev)]
        r = list(edit(ev, "separate", [1]))
        m = ev.mesh()
        d = np.linalg.norm(m.delta([0, 0, 1], m.vertices), axis=1)
        a, b = np.argsort(d)[:2]                         # (the two copies)
        r += list(edit(ev, "snap", [m.vertex_ids[a], m.vertex_ids[b]]))
        return r
    a, b = _both(_tetras, act)
    assert list(a["result"]) == [0, 0] and a["problems"] == ["a pinched vertex"]
    _same(a, b)


def _wetting(periods):
    """A cap on the floor on the cell's corner, its first ring in from the
    contact line pressed down onto the floor."""
    cap = shapes.cap(0.5, 0.25, nr=6, nt=24)
    v, f = np.asarray(cap.vertices, float), np.asarray(cap.faces)
    rim = np.asarray(cap.rims["rim"])
    ring = sorted({int(b) for t in f for a, b in zip(t, np.roll(t, 1))
                   if a in rim and b not in rim})
    v[ring, 2] = 1e-3
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), f,
                   constraints={"bottom": C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
                   vertex_constraints={"bottom": rim}, bodies=[pyse.Body()], periods=periods)
    return ev


def test_patch_on_the_cells_corner():
    def act(ev):
        m = ev.mesh()
        low = np.flatnonzero((m.vertices[np.asarray(m.facets), 2] < 2e-3).all(axis=1))
        assert len(low) == 48
        return edit(ev, "patch", [walls(ev)[0][0]] + list(np.asarray(m.facet_ids)[low]))
    a, b = _both(_wetting, act)
    assert list(a["result"]) == [0] and a["problems"] == []
    _same(a, b)


def _fans(periods):
    """Two fans on the floor joined along an edge across the cell's corner
    (test_topology's unzip, its wall the floor)."""
    v = np.array([[0, 0, 0], [0.2, 0, 0], [-0.1, 0, 0.15], [0.1, 0.15, 0.15], [0.3, 0, 0.15],
                  [-0.1, 0, -0.15], [0.1, 0.15, -0.15], [0.3, 0, -0.15]], float)
    v = v[:, [0, 2, 1]] - [0.1, 0, 0]
    f = np.array([[0, 1, 3], [0, 3, 2], [1, 4, 3], [1, 0, 6], [0, 5, 6], [1, 6, 7]])[:, ::-1]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), f,
                   constraints={"bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0)},
                   vertex_constraints={"bottom": [0, 1, 2, 4, 5, 7]}, periods=periods)
    return ev


def test_unzip_on_the_cells_corner():
    def act(ev):
        m = ev.mesh()
        joint = [k for k, (a, b) in enumerate(np.asarray(m.edges)) if {int(a), int(b)} == {0, 1}]
        return edit(ev, "unzip", [int(m.edge_ids[joint[0]])])
    a, b = _both(_fans, act)
    assert list(a["result"]) == [0] and len(a["mesh"].vertices) == 10
    _same(a, b)


def _on_ball(periods):
    """A drop on a ball whose top is on the cell's corner."""
    cap = shapes.cap(0.5, 0.25, nr=6, nt=24)
    v = cap.vertices + [0, 0, 1.0]
    rim = np.asarray(cap.rims["rim"])
    v[rim] /= np.linalg.norm(v[rim], axis=1)[:, None]
    ev = pyse.Evolver()
    ev.load_arrays(_into_cell(v, periods), cap.faces,
                   constraints={"ball": C.sphere((0, 0, 0), 1.0, contact_angle=60.0)},
                   vertex_constraints={"ball": rim}, bodies=[pyse.Body(volume=None)],
                   periods=periods)
    ev.body(1).target = ev.body(1).volume
    return ev


def test_contact_lines_chopped_across_the_corner():
    # (the lines' edges' lengths at the nearest images: a raw length, a
    # period, split every step)
    def act(ev):
        m = ev.mesh()
        line = m.edge_valence() == 1
        was = _core.line_chop(0.6*float(m.edge_lengths()[line].min()))
        try:
            ev.command("g 1")
        finally:
            _core.line_chop(was)
        return [int((ev.mesh().edge_valence() == 1).sum())]
    a, b = _both(lambda p: _on_ball(None if p is None else (2.5, 2.7)), act)
    assert a["result"][0] > 24
    _same(a, b, tol=1e-11)


def test_a_torus_not_a_slab_is_still_refused():
    ev = pyse.Evolver()
    ev.load_string("TORUS\nperiods\n1 0 0\n0 1 0\n0 0 1\nvertices\n1 0 0 0\n2 0.1 0 0\n"
                   "3 0 0.1 0\nedges\n1 1 2 * * *\n2 2 3 * * *\n3 3 1 * * *\nfaces\n1 1 2 3\n")
    assert ev.torus and ev.periods is None
    for call in (lambda: crossings(ev), lambda: check_topology(ev),
                 lambda: edit(ev, "smooth", [1]), lambda: remesh(ev, 0.1)):
        with pytest.raises(pyse.EvolverError, match="torus"):
            call()
