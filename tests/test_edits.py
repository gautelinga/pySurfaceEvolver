"""The checked local mesh edits (bindings/pyse_edit.c, pysurfaceevolver._edits)."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes
from pysurfaceevolver._edits import REASONS, crossings, edit
from pysurfaceevolver._surgery import check_topology

warnings.simplefilter("ignore")


def _problems(ev):
    return {p[1].split(" (")[0] for p in check_topology(ev)}


def _tally(results):
    values, counts = np.unique(results, return_counts=True)
    return {REASONS[int(v)]: int(n) for v, n in zip(values, counts)}


def _drop():
    # a 60-degree drop on a unit sphere (a wall)
    cap = shapes.cap(0.5, 0.25, nr=6, nt=24)
    v = cap.vertices + [0, 0, 1.0]
    rim = np.asarray(cap.rims["rim"])
    v[rim] /= np.linalg.norm(v[rim], axis=1)[:, None]
    ev = pyse.Evolver()
    ev.load_arrays(v, cap.faces, constraints={"ball": C.sphere((0, 0, 0), 1.0, contact_angle=60.0)},
                   vertex_constraints={"ball": rim}, bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    ev.relax()
    return ev


def _edge(ev, a, b):
    m = ev.mesh()
    e = np.asarray(m.edges)
    return int(np.asarray(m.edge_ids)[[k for k in range(len(e)) if set(e[k]) == {a, b}][0]])


def test_collapse_split_and_flip_keep_the_surface_sound():
    ev = _drop()
    m = ev.mesh()
    facets = ev.counts["facets"]
    shortest = np.asarray(m.edge_ids)[np.argsort(m.edge_lengths())[:15]]
    r = edit(ev, "collapse", shortest)
    assert (r == 0).sum() >= 10 and ev.counts["facets"] == facets - 2*(r == 0).sum()
    assert _problems(ev) == set() and ev.check() == []
    m = ev.mesh()
    longest = np.asarray(m.edge_ids)[np.argsort(-m.edge_lengths())[:10]]
    r = edit(ev, "split", longest)
    assert (r == 0).all() and _problems(ev) == set() and ev.check() == []
    m = ev.mesh()
    r = edit(ev, "flip", np.asarray(m.edge_ids)[m.edge_valence() == 2], mode=2)
    assert (r == 0).any() and _problems(ev) == set() and ev.check() == []
    ev.relax()


def test_a_contact_lines_vertex_collapses_only_along_the_line():
    ev = _drop()
    m = ev.mesh()
    on = ev.on_constraint("ball")
    e = np.asarray(m.edges)
    inward = [k for k in range(len(e)) if on[e[k]].sum() == 1]
    r = edit(ev, "collapse", np.asarray(m.edge_ids)[inward[:5]], max_turn=89, max_volume=1)
    # the free end goes onto the line; never the line's vertex inward
    assert (r == 0).any()
    assert _problems(ev) == set()
    assert ev.on_constraint("ball").sum() == on.sum()


def test_smoothing_evens_a_jiggled_mesh():
    ev = _drop()
    x = ev.vertices.copy()
    free = ~ev.on_constraint("ball")
    x[free] += 0.01*np.random.default_rng(0).standard_normal((free.sum(), 3))
    ev.vertices = x
    before = ev.mesh_quality().angle_min
    for _ in range(3):
        edit(ev, "smooth", ev.mesh().vertex_ids)
    assert ev.mesh_quality().angle_min > before + 5
    assert _problems(ev) == set()


def test_separate_and_snap_undo_each_other():
    tv = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    tf = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[tv, -tv[1:]], np.r_[tf, np.where(tf == 0, 0, tf + 3)[:, ::-1]],
                   bodies=[pyse.Body(faces=range(4)), pyse.Body(faces=range(4, 8))])
    assert _problems(ev) == {"a pinched vertex"}
    assert list(edit(ev, "separate", [1])) == [0]
    assert _problems(ev) == set() and ev.counts["vertices"] == 8 and ev.check() == []
    x = np.asarray(ev.mesh().vertices)
    a, b = np.argsort(np.linalg.norm(x, axis=1))[:2]        # the two copies
    ids = np.asarray(ev.mesh().vertex_ids)
    assert list(edit(ev, "snap", [ids[a], ids[b]])) == [0]
    assert _problems(ev) == {"a pinched vertex"} and ev.counts["vertices"] == 7


def test_a_degenerate_facet_goes():
    box = shapes.box(n=4)
    ev = pyse.Evolver()
    ev.load_arrays(box.vertices, box.faces, bodies=[pyse.Body()])
    x = ev.vertices.copy()
    f = np.asarray(ev.mesh().facets)
    k = [i for i in range(len(f)) if np.all(np.abs(x[f[i], 2] - 1) < 1e-9)
         and np.all((x[f[i], :2] > 0.01) & (x[f[i], :2] < 0.99))][0]
    a, b, c = f[k]
    x[a] = 0.98*(x[b] + x[c])/2 + 0.02*x[a]
    ev.vertices = x
    facets = ev.counts["facets"]
    assert list(edit(ev, "degenerate", [ev.mesh().facet_ids[k]], flat=0.05)) == [0]
    assert ev.counts["facets"] <= facets and _problems(ev) == set() and ev.check() == []


def test_an_edit_that_makes_a_crossing_is_undone_and_the_batch_kept():
    # sheet A: a ridge whose flip would pass under sheet B, and a flat quad
    # beside it whose flip is harmless; B: a triangle just above A's middle
    v = np.array([(0, 0, 0.2), (1, 1, 0.2), (1, 0, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]
                 + [(0.5 + 0.3*np.cos(t), 0.5 + 0.3*np.sin(t), 0.05)
                    for t in np.radians([45, 165, 285])], float)
    f = np.array([(0, 2, 1), (1, 3, 0), (2, 4, 5), (2, 5, 1), (6, 7, 8)])
    ev = pyse.Evolver()
    ev.load_arrays(v, f)
    assert crossings(ev) == 0
    r = edit(ev, "flip", [_edge(ev, 2, 5), _edge(ev, 0, 1)], max_turn=89, max_volume=1)
    assert [REASONS[int(k)] for k in r] == ["done", "an edge through a facet"]
    assert crossings(ev) == 0
    e = [set(x) for x in np.asarray(ev.mesh().edges)]
    assert {4, 1} in e and {2, 3} not in e and {0, 1} in e


def test_crossings_are_counted():
    box = shapes.box(n=4)
    v, f = np.asarray(box.vertices), np.asarray(box.faces)
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[v, v + [0.51, 0.33, 0.27]], np.r_[f, f + len(v)],
                   bodies=[pyse.Body(faces=range(len(f))), pyse.Body(faces=range(len(f), 2*len(f)))])
    assert crossings(ev) > 0
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[v, v + [1.5, 0, 0]], np.r_[f, f + len(v)],
                   bodies=[pyse.Body(faces=range(len(f))), pyse.Body(faces=range(len(f), 2*len(f)))])
    assert crossings(ev) == 0


def test_edit_refuses_unknown_parameters():
    ev = _drop()
    with pytest.raises(TypeError):
        edit(ev, "flip", [1], turn=3)
