"""Topology changes done with Evolver's own commands (see docs/PLAN.md,
"Future: topology changes")."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, recipes, shapes


def _pieces(ev):
    """The surface's edge-connected pieces, each as (the volume it encloses
    with the plane z = z0 of its rim, z0 = 0 or 3, the body of its facets)."""
    m = ev.mesh()
    f, x = np.asarray(m.facets), ev.vertices
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    _, edge = np.unique(e, axis=0, return_inverse=True)
    edge = edge.ravel().reshape(3, -1).T
    label = np.arange(len(f))                 # facets sharing an edge, by relabelling
    while True:
        low = np.full(edge.max() + 1, len(f))
        np.minimum.at(low, edge.ravel(), np.repeat(label, 3))
        new = low[edge].min(axis=1)
        if np.array_equal(new, label):
            break
        label = new
    out = []
    for c in np.unique(label):
        ff = f[label == c]
        z0 = 0.0 if x[ff, 2].min() < 0.5 else 3.0
        a, b, d = (x[ff[:, i]] - [0, 0, z0] for i in range(3))
        body = np.unique(m.facet_bodies[label == c])
        out.append((np.einsum("ij,ij->i", a, np.cross(b, d)).sum()/6, z0, body[body > 0]))
    return out


def _border(ev):
    """The edges (vertex rows) with one facet."""
    f = np.asarray(ev.mesh().facets)
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    u, count = np.unique(e, axis=0, return_counts=True)
    return u[count == 1]


def _merge(ev):
    """Glue bodies 1 and 2 where they come closest (their closest facets) and
    make them one body, 1."""
    m = ev.mesh()
    centre = ev.vertices[np.asarray(m.facets)].mean(axis=1)
    body = np.asarray(m.facet_bodies).max(axis=1)
    a = np.where(body == 1)[0][np.argmax(centre[body == 1, 0])]
    b = np.where(body == 2)[0][np.argmin(centre[body == 2, 0])]
    fa, fb = int(m.facet_ids[a]), int(m.facet_ids[b])
    for command in (f"facet_merge({fa},{fb})", f"unset facet[{fa}] frontbody",
                    f"dissolve facet[{fa}]", "dissolve edges; dissolve vertices",
                    "set facet frontbody 1 where frontbody == 2",
                    "set facet backbody 1 where backbody == 2", "rebody", "dissolve body[2]"):
        ev.command(command)


def test_a_bridge_that_pinches_off_splits_into_two_drops_with_their_volumes():
    # a bridge between disks 3 apart, pinned at their rims (the top one closes
    # the body through its content integral), loses liquid until it pinches:
    # Evolver's tiny-edge deletion shrinks the neck to a vertex, "pop vertex"
    # splits it, and "rebody" makes the two pieces bodies, each holding its
    # volume. The top rim's content belongs to the upper drop after that, also
    # when the rim is refined
    H, nt = 3.0, 16
    tube = shapes.tube(1.0, 0.0, H, nz=8, nt=nt)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   boundaries={1: C.circle((0, 0, 0)),
                               2: C.circle((0, 0, H), content=(f"-{H}*y/2", f"{H}*x/2", "0"))},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=np.pi*H)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ev.relax().converged
        full = ev.body(1).volume
        v = 1.0
        for _ in range(100):                      # 2% of the liquid at a time
            if len(_pieces(ev)) == 2:
                break
            x = ev.vertices
            middle = np.abs(x[:, 2] - H/2) < 0.9
            if np.hypot(x[middle, 0], x[middle, 1]).min() > 0.3:
                v -= 0.02
                ev.body(1).target = v*full
            ev.command("g 10")
            ev.remesh(target=0.25)
        assert len(_pieces(ev)) == 2 and 0.3 < v < 0.6
        before = ev.body(1).volume
        ev.command("pop vertex")
        ev.command("rebody")
        assert len(ev.bodies().ids) == 2
        assert ev.bodies().volume.sum() == pytest.approx(before, rel=1e-12)
        for volume, z0, body in _pieces(ev):
            assert len(body) == 1
            assert ev.body(int(body[0])).volume == pytest.approx(volume, rel=1e-10)
            assert ev.body(int(body[0])).target == pytest.approx(volume, rel=1e-10)
        assert ev.relax(levels=1).converged
    for volume, z0, body in _pieces(ev):
        assert ev.body(int(body[0])).volume == pytest.approx(volume, rel=1e-8)
        # the drop is a cap on the rim: its pressure is 4h/(1 + h^2)
        h = np.roots([np.pi/6, 0, np.pi/2, -volume]).real.max()
        assert ev.body(int(body[0])).pressure == pytest.approx(4*h/(1 + h*h), rel=0.02)


def test_two_drops_that_touch_merge_into_one_with_both_volumes():
    # two spheres 0.02 apart: facet_merge glues the closest facets, dissolving
    # the glued facet opens the neck, and the second body's facets go to the
    # first (rebody's own merge only joins bodies on the two sides of a facet);
    # rebody then rebuilds the bodies' quantities (not needed here, where no
    # wire carries a content integral, as in the pinch-off). The drop holding
    # both volumes relaxes to the sphere
    gap = 0.02
    t = np.linspace(0, np.pi, 9)
    ball = shapes.revolve(np.sin(t), -np.cos(t), 16)
    n, nf = len(ball.vertices), len(ball.faces)
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[ball.vertices - [1 + gap/2, 0, 0], ball.vertices + [1 + gap/2, 0, 0]],
                   np.r_[ball.faces, ball.faces + n],
                   bodies=[pyse.Body(faces=range(nf), volume=None),
                           pyse.Body(faces=range(nf, 2*nf), volume=None)])
    for i in (1, 2):
        ev.body(i).target = ev.body(i).volume
    total = ev.bodies().volume.sum()
    _merge(ev)
    assert len(_pieces(ev)) == 1 and list(ev.bodies().ids) == [1]
    assert len(_border(ev)) == 0                  # closed (and no edge with three facets:)
    f = np.asarray(ev.mesh().facets)
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    assert np.unique(e, axis=0, return_counts=True)[1].max() == 2
    assert ev.body(1).volume == pytest.approx(total, rel=0.01)   # and the glued gap's
    ev.body(1).target = total
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(60):                       # the neck widens: keep its edges even
            ev.command("g 5")
            ev.remesh(target=0.35)
        assert ev.relax(levels=1).converged
    R = (3*total/(4*np.pi))**(1/3)
    assert ev.body(1).volume == pytest.approx(total, rel=1e-8)
    assert ev.energy == pytest.approx(4*np.pi*R*R, rel=0.005)
    assert ev.body(1).pressure == pytest.approx(2/R, rel=0.005)


def test_drops_merged_on_a_plane_wet_it_between_them():
    # two 120-degree drops on a plane, merged where they touch above it: the
    # neck sinks onto the plane, and where it gets there the facets under it
    # go and their border becomes contact line on the plane's constraint. Its
    # wetting energy is a line integral along the edges as their facets go
    # round them, and Evolver takes each edge's integral along the edge as
    # stored, with the sign its orientation has when the edge is put on the
    # constraint: so "orientation -1" first on an edge stored the other way
    # (reverse_orientation keeps the integral as it was, and setting the
    # orientation after doesn't change it). The two contact lines become one,
    # and the drop relaxes to the cap
    theta, R, gap = 120.0, 0.84, 0.02
    t = np.radians(theta)
    drop = shapes.cap(R, R*(1 - np.cos(t)), nr=6, nt=24)
    n, nf = len(drop.vertices), len(drop.faces)
    rim = np.asarray(drop.rims["rim"])
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[drop.vertices - [R + gap/2, 0, 0], drop.vertices + [R + gap/2, 0, 0]],
                   np.r_[drop.faces, drop.faces + n],
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=theta)},
                   vertex_constraints={1: np.r_[rim, rim + n]},
                   bodies=[pyse.Body(faces=range(nf), volume=None),
                           pyse.Body(faces=range(nf, 2*nf), volume=None)])
    for i in (1, 2):
        ev.body(i).target = ev.body(i).volume

    def wetting_energy():
        f, x = np.asarray(ev.mesh().facets), ev.vertices
        a = 0.5*np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
        return np.linalg.norm(a, axis=1).sum() - np.cos(t)*abs(a[:, 2].sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ev.relax().converged
        total = ev.bodies().volume.sum()
        _merge(ev)
        ev.body(1).target = total
        for i in range(100):                      # until the neck reaches the plane
            ev.command("g 2")
            if i % 3 == 0:
                ev.remesh(target=0.2)
            free = ~np.asarray(ev.on_constraint(1))
            if ev.vertices[free, 2].min() < 0.01:
                break
        assert ev.vertices[free, 2].min() < 0.01
        assert len(_pieces(ev)) == 1
        m = ev.mesh()
        f, x = np.asarray(m.facets), ev.vertices
        c = x[f].mean(axis=1)
        down = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])[:, 2] < 0
        under = (c[:, 2] < 0.12) & (np.abs(c[:, 0]) < 0.3) & down
        ev.command("; ".join(f"dissolve facet[{int(i)}]" for i in m.facet_ids[under]))
        ev.command("dissolve edges; dissolve vertices")
        m = ev.mesh()
        f, edges = np.asarray(m.facets), np.asarray(m.edges)
        along = {(r[i], r[(i + 1) % 3]) for r in f for i in range(3)}
        row = {tuple(sorted(e)): i for i, e in enumerate(edges)}
        on = np.asarray(ev.on_constraint(1))
        for a, b in _border(ev):
            for v in (a, b):
                if not on[v]:
                    ev.command(f"set vertex[{int(m.vertex_ids[v])}] constraint 1")
                    on[v] = True
            i = row[(a, b)]
            e = int(m.edge_ids[i])
            if not ev.eval(f"edge[{e}].on_constraint 1"):
                if tuple(edges[i]) not in along:
                    ev.command(f"set edge[{e}] orientation -1")
                ev.command(f"set edge[{e}] constraint 1")
        assert len(_border(ev)) > 0 and _loops(_border(ev)) == 1
        assert ev.energy == pytest.approx(wetting_energy(), rel=1e-10)
        for _ in range(60):
            ev.command("g 5")
            ev.remesh(target=0.2)
        assert ev.relax(levels=1).converged
    assert ev.body(1).volume == pytest.approx(total, rel=1e-8)
    assert ev.energy == pytest.approx(wetting_energy(), rel=1e-10)
    Rs = (total/(np.pi*(2 - 3*np.cos(t) + np.cos(t)**3)/3))**(1/3)
    h, a = Rs*(1 - np.cos(t)), Rs*np.sin(t)
    assert ev.energy == pytest.approx(2*np.pi*Rs*h - np.cos(t)*np.pi*a*a, rel=0.002)
    x = ev.vertices[np.asarray(ev.on_constraint(1))]
    assert np.hypot(x[:, 0] - x[:, 0].mean(), x[:, 1] - x[:, 1].mean()).mean() == \
        pytest.approx(a, rel=0.01)


def _loops(edges):
    """How many closed loops the edges make."""
    parent = {}
    def root(v):
        while parent.setdefault(v, v) != v:
            v = parent[v]
        return v
    for a, b in edges:
        parent[root(a)] = root(b)
    return len({root(v) for v in parent})


# ---- stage A: events coming (Evolver.events, and runs stopping at them) ----------

def _two_spheres(gap):
    t = np.linspace(0, np.pi, 9)
    ball = shapes.revolve(np.sin(t), -np.cos(t), 16)
    n, nf = len(ball.vertices), len(ball.faces)
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[ball.vertices - [1 + gap/2, 0, 0], ball.vertices + [1 + gap/2, 0, 0]],
                   np.r_[ball.faces, ball.faces + n],
                   bodies=[pyse.Body(faces=range(nf), volume=None),
                           pyse.Body(faces=range(nf, 2*nf), volume=None)])
    return ev


def _two_drops(theta, gap, R=0.84):
    """Two drops on the plane z = 0 at contact angle theta, their contact
    lines gap apart (theta <= 90) or their equators (theta > 90)."""
    t = np.radians(theta)
    drop = shapes.cap(R, R*(1 - np.cos(t)), nr=6, nt=24)
    n, nf = len(drop.vertices), len(drop.faces)
    rim = np.asarray(drop.rims["rim"])
    s = (R*np.sin(t) if theta <= 90 else R) + gap/2
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[drop.vertices - [s, 0, 0], drop.vertices + [s, 0, 0]],
                   np.r_[drop.faces, drop.faces + n],
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=theta)},
                   vertex_constraints={1: np.r_[rim, rim + n]},
                   bodies=[pyse.Body(faces=range(nf), volume=None),
                           pyse.Body(faces=range(nf, 2*nf), volume=None)])
    return ev


def test_two_spheres_about_to_touch_are_an_event():
    assert _two_spheres(0.5).events() == []
    found = _two_spheres(0.02).events()
    assert [e.kind for e in found] == ["touch"]
    assert found[0].gap == pytest.approx(0.02, abs=1e-9)
    assert np.linalg.norm(found[0].where) < 0.05


def test_contact_lines_about_to_meet_are_an_event():
    assert _two_drops(60.0, 0.3).events() == []
    found = [e for e in _two_drops(60.0, 0.02).events() if e.kind == "contact lines"]
    assert len(found) == 1 and found[0].constraint == 1
    assert found[0].gap == pytest.approx(0.02, abs=1e-9)


def test_a_thinning_bridge_is_a_neck_before_it_pinches():
    # the bridge between disks of the pinch-off test, losing liquid: the neck
    # is an event while it is still a ring of about five edges, before
    # Evolver's tiny-edge deletion closes it (here at the last look before:
    # each look is 10 gradient steps and a remesh apart)
    H, nt = 3.0, 16
    tube = shapes.tube(1.0, 0.0, H, nz=8, nt=nt)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   boundaries={1: C.circle((0, 0, 0)),
                               2: C.circle((0, 0, H), content=(f"-{H}*y/2", f"{H}*x/2", "0"))},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=np.pi*H)])
    first = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.relax()
        full, v = ev.body(1).volume, 1.0
        for i in range(100):
            if len(_pieces(ev)) == 2:
                break
            necks = [e for e in ev.events() if e.kind == "neck"]
            if necks and first is None:
                first = (i, necks[0])
            x = ev.vertices
            middle = np.abs(x[:, 2] - H/2) < 0.9
            if np.hypot(x[middle, 0], x[middle, 1]).min() > 0.3:
                v -= 0.02
                ev.body(1).target = v*full
            ev.command("g 10")
            ev.remesh(target=0.25)
    assert first is not None and first[0] < i
    assert abs(first[1].where[2] - H/2) < 0.5 and first[1].gap > 0.2


def _drop_under_a_sphere():
    """A hemispherical drop on the plane z = 0 (90 degrees, radius 0.7) under
    a solid sphere of radius 1 whose bottom is at height 1."""
    drop = shapes.cap(0.7, 0.7, nr=6, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(drop.vertices, drop.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                                2: C.sphere((0, 0, 2.0), 1.0, contact_angle=90.0)},
                   vertex_constraints={1: drop.rims["rim"]},
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    return ev


def test_arclength_stops_before_a_growing_drop_touches_a_solid():
    ev = _drop_under_a_sphere()
    with pytest.warns(UserWarning, match="stopped at .* constraint 2"):
        steps = list(recipes.arclength(ev, ev.body(1), 0.1, 60))
    last = steps[-1]
    assert [e.kind for e in last.events] == ["solid"] and last.events[0].between == "outside"
    assert 0 <= last.events[0].gap < 0.5*last.events[0].scale
    assert steps[-2].events and all(not s.events for s in steps[:-2])     # closer at the last
    top = ev.vertices[:, 2].max()
    assert 0.9 < top < 1.0                  # below the sphere's bottom, close to it


def test_continuation_stops_before_a_growing_drop_touches_a_solid():
    ev = _drop_under_a_sphere()
    v0 = ev.body(1).volume
    with pytest.warns(UserWarning, match="stopped at .* constraint 2"):
        steps = list(recipes.continuation(ev, ev.body(1), v0*np.linspace(1.1, 4.0, 30)))
    assert steps[-1].events and steps[-1].events[0].kind == "solid"
    assert ev.vertices[:, 2].max() < 1.0
    assert len(steps) < 30
