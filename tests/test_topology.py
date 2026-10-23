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
    # below the sphere's bottom, within the event's reach of it (the apex's
    # own edge length)
    assert 0 < 1.0 - top < last.events[0].scale


def test_continuation_stops_before_a_growing_drop_touches_a_solid():
    ev = _drop_under_a_sphere()
    v0 = ev.body(1).volume
    with pytest.warns(UserWarning, match="stopped at .* constraint 2"):
        steps = list(recipes.continuation(ev, ev.body(1), v0*np.linspace(1.1, 4.0, 30)))
    assert steps[-1].events and steps[-1].events[0].kind == "solid"
    assert ev.vertices[:, 2].max() < 1.0
    assert len(steps) < 30


# ---- stage C: pinching off (Evolver.pinch_off, arclength(topology=True)) ---------

def _sphere_bridge(volume):
    """bridge_rupture.md's bridge: between unit spheres 0.6 apart at 40 degrees,
    2048 facets."""
    c, a = 1.3, 0.35
    zb = c - np.sqrt(1 - a*a)
    tube = shapes.tube(a, -zb, zb, nz=4, nt=16)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   constraints={1: C.sphere((0, 0, -c), 1.0, contact_angle=40.0,
                                            wet_poles=("north",)),
                                2: C.sphere((0, 0, c), 1.0, contact_angle=40.0,
                                            wet_poles=("south",))},
                   vertex_constraints={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=volume)])
    ev.relax(levels=2)
    return ev


def _cap_on_sphere(volume, theta):
    """The pressure of a drop of this volume on a unit sphere at contact angle
    theta (degrees): a sphere of radius r cutting it at that angle."""
    t = np.radians(theta)

    def held(r):
        d = np.sqrt(1 + r*r - 2*r*np.cos(t))
        lens = np.pi*(1 + r - d)**2*(d*d + 2*d*r - 3*r*r + 2*d + 6*r - 3)/(12*d)
        return 4*np.pi*r**3/3 - lens
    lo, hi = 1e-3, 10.0
    for _ in range(200):
        mid = (lo + hi)/2
        lo, hi = (mid, hi) if held(mid) < volume else (lo, mid)
    return 2/mid


def test_a_shrinking_bridge_between_spheres_falls_into_two_drops():
    # the bridge turns unstable at a branch point (0.0790 exact, under 0.0789
    # here), its liquid moving to one sphere. With topology=True arclength
    # falls from the first unstable point: the neck comes and is cut, and each
    # piece relaxes to a drop on its sphere (an exact cap), the two holding
    # what the bridge did. The run goes on draining the larger, body 1; the
    # smaller keeps its volume
    ev = _sphere_bridge(0.09)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steps = list(recipes.arclength(ev, ev.body(1), -0.003, 12, topology=True))
    k = [i for i, s in enumerate(steps) if s.fell]
    assert len(k) == 1 and [s.index for s in steps] == list(range(len(steps)))
    fell, lost = steps[k[0]], steps[k[0] - 1]
    assert lost.result.negative_modes == 1 and steps[k[0] - 2].result.negative_modes == 0
    assert 0.076 < lost.value < 0.0789
    assert [e.kind for e in fell.changes] == ["neck"]
    assert fell.volumes.sum() == pytest.approx(lost.value, rel=1e-8)
    assert fell.volumes[0] > fell.volumes[1] and 0.1 < fell.volumes[1]/lost.value < 0.3
    for v, p in zip(fell.volumes, fell.pressures):
        assert p == pytest.approx(_cap_on_sphere(v, 40.0), rel=0.005)
    after = steps[k[0] + 1:]
    assert len(after) >= 2 and all(not s.fell for s in after)
    assert after[-1].value < fell.value - 0.003
    for s in after:
        assert s.volumes[1] == pytest.approx(fell.volumes[1], rel=1e-8)
        assert s.result.negative_modes == 0


def _backed_up_run(monkeypatch, caplog, fail, steps=14):
    """The shrinking bridge's run (above), its descents' first ``fail``
    failing (_surgery.descend, stopping as one that ran away), or with
    fail=-1 the first neck's cut raising (as an edit failing in Evolver
    would): the steps (numbered on from 1000 as they come, as a resumed
    run's caller numbers them), the warnings, the backing up logged."""
    import logging
    from pysurfaceevolver import _surgery
    descend, pinch_off, calls = _surgery.descend, _surgery.pinch_off, []

    def failing_descend(*args, **kwargs):
        calls.append("descend")
        if calls.count("descend") <= fail:
            return None, [], "the descent ran away (a test's)"
        return descend(*args, **kwargs)

    def failing_pinch_off(*args, **kwargs):
        calls.append("cut")
        if fail == -1 and calls.count("cut") == 1:
            raise RuntimeError("a test's edit failing")
        return pinch_off(*args, **kwargs)
    monkeypatch.setattr(_surgery, "descend", failing_descend)
    monkeypatch.setattr(_surgery, "pinch_off", failing_pinch_off)
    ev = _sphere_bridge(0.09)
    with warnings.catch_warnings(record=True) as caught, \
            caplog.at_level(logging.DEBUG, logger="pysurfaceevolver.topology"):
        warnings.simplefilter("always")
        run = []
        for step in recipes.arclength(ev, ev.body(1), -0.003, steps, topology=True):
            step.index += 1000
            run.append(step)
    stops = [str(w.message) for w in caught if "arclength()" in str(w.message)]
    return run, stops, [r.getMessage() for r in caplog.records if "branch" in r.getMessage()]


@pytest.mark.parametrize("fail", [1, -1])
def test_arclength_backs_up_where_a_descent_fails_and_goes_on(monkeypatch, caplog, fail):
    # the fall from the first unstable point failing (its descent stopping,
    # or an edit raising in it), the run backs up to the last stable point
    # and goes on from there at half the step: it loses its stability
    # nearer the branch point, falls, the neck is cut, the pieces are drops
    # as without the failure, and past where it backed up the step is ds
    # again
    steps, stops, said = _backed_up_run(monkeypatch, caplog, fail)
    assert not stops
    assert said[0].startswith("branch backed up 1 step: the fall from 0.0775")
    assert ("the descent ran away (a test's)" if fail == 1 else
            "RuntimeError: a test's edit failing") in said[0] and "ds -0.0015" in said[0]
    assert said[1].startswith("branch past where it backed up: ds -0.003")
    k = [i for i, s in enumerate(steps) if s.fell]
    assert len(k) == 1
    fell, lost, again = steps[k[0]], steps[k[0] - 1], steps[k[0] - 2]
    assert lost.index == again.index and lost.result.negative_modes == 1
    assert 0.0775 < again.value < lost.value < 0.0789     # (half a step from the last stable)
    assert [e.kind for e in fell.changes] == ["neck"]
    assert fell.volumes.sum() == pytest.approx(lost.value, rel=1e-8)
    assert 0.1 < fell.volumes[1]/lost.value < 0.3
    for v, p in zip(fell.volumes, fell.pressures):
        assert p == pytest.approx(_cap_on_sphere(v, 40.0), rel=0.005)
    assert all(s.result.negative_modes == 0 for s in steps[k[0] + 1:])


def test_arclength_stops_where_its_tries_run_out(monkeypatch, caplog):
    # every fall failing: four tries at a place, each further back (and on
    # at a new place once past it, twelve in a run at most), then the run
    # stops as without them
    steps, stops, said = _backed_up_run(monkeypatch, caplog, 100, steps=60)
    assert len(stops) == 1 and "stopped: the descent ran away (a test's)" in stops[0]
    backed = [s for s in said if s.startswith("branch backed up")]
    assert 4 <= len(backed) <= 12 and said[-1].startswith("branch not backed up again")
    back = [0 if s.split()[3] == "to" else int(s.split()[3]) for s in backed[:4]]
    assert back == sorted(back)                     # (each further back)
    assert not any(s.fell for s in steps)


def test_a_descent_past_its_budget_stops_and_the_run_backs_up(monkeypatch, caplog):
    # descents bounded at a pass over the surface: each stops at its first
    # iteration, the run backs up as from any stop and ends where its tries
    # run out, with the reason
    import logging
    from pysurfaceevolver import _surgery
    monkeypatch.setattr(_surgery, "_BUDGET", 1)
    ev = _sphere_bridge(0.09)
    with warnings.catch_warnings(record=True) as caught, \
            caplog.at_level(logging.DEBUG, logger="pysurfaceevolver.topology"):
        warnings.simplefilter("always")
        steps = list(recipes.arclength(ev, ev.body(1), -0.003, 60, topology=True))
    stops = [str(w.message) for w in caught if "arclength()" in str(w.message)]
    assert len(stops) == 1 and "the descent's budget spent" in stops[0]
    assert any(r.getMessage().startswith("branch backed up") for r in caplog.records)
    assert not any(s.fell for s in steps)


def test_a_run_past_its_budget_stops_after_its_last_step():
    # the run's work bounded: it ends with a warning after the step that
    # spent it, at the same step every time (work, not time); the caller's
    # work between steps isn't counted
    def run(budget, between=False):
        ev = _sphere_bridge(0.09)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = []
            for s in recipes.arclength(ev, ev.body(1), -0.003, 14, topology=True,
                                       budget=budget):
                out.append(s)
                if between:
                    ev.relax()
        return out, [str(w.message) for w in caught if "arclength()" in str(w.message)]
    full, stops = run(None)
    assert not stops and any(s.fell for s in full)
    a, stops = run(200)
    assert len(stops) == 1 and "the run's budget spent" in stops[0]
    assert 2 <= len(a) < len(full)
    b, _ = run(200)
    c, _ = run(200, between=True)
    # (the same steps; their values to rounding: OpenMP's sums, and the
    # caller's relax() moving the surface by that much)
    assert len(a) == len(b) == len(c)
    for other in (b, c):
        assert [s.value for s in other] == pytest.approx([s.value for s in a], rel=1e-9)


def test_a_bridge_between_disks_pinches_at_its_fold_into_two_equal_drops():
    # pinned to the rims of disks 3 apart (slenderness 1.5): stable down to
    # the fold, volume 0.4751 of the cylinder (disk_bridge.md), where the neck
    # thins in the middle. Past the fold (one step past the last stable point)
    # it falls and pinches into two equal caps on the rims; the upper keeps
    # the top rim's content integral (Evolver's rebody, fixed)
    H = 3.0
    tube = shapes.tube(1.0, 0.0, H, nz=8, nt=16)
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces,
                   boundaries={1: C.circle((0, 0, 0)),
                               2: C.circle((0, 0, H), content=(f"-{H}*y/2", f"{H}*x/2", "0"))},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   bodies=[pyse.Body(volume=0.55*np.pi*H)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.relax(levels=1)
        steps = list(recipes.arclength(ev, ev.body(1), -0.1, 20, topology=True))
    k = [i for i, s in enumerate(steps) if s.fell][0]
    fell, (previous, stable, lost) = steps[k], steps[k - 3:k]
    assert lost.value > stable.value                          # turned at the fold
    assert stable.value/(np.pi*H) == pytest.approx(0.4751, rel=0.02)
    total = min(stable.value, lost.value) + (stable.value - previous.value)
    assert fell.volumes.sum() == pytest.approx(total, rel=1e-8)
    assert abs(fell.volumes[0] - fell.volumes[1]) < 0.01*total
    for volume, pressure in zip(fell.volumes, fell.pressures):
        h = np.roots([np.pi/6, 0, np.pi/2, -volume]).real.max()
        assert pressure == pytest.approx(4*h/(1 + h*h), rel=0.01)


def test_a_catenoid_pulled_past_its_fold_falls_into_two_disks():
    # a soap film between wires, no bodies, the run in a parameter (the
    # wires' distance): past the fold (1.3255) it falls and pinches, and
    # the two pieces are flat disks spanning the wires
    tube = shapes.tube(1.0, 0.0, 0.6, nz=4, nt=16)
    top = C.curve("cos(p1)", "sin(p1)", "ht", lambda x: np.arctan2(x[:, 1], x[:, 0]))
    ev = pyse.Evolver()
    ev.load_arrays(tube.vertices, tube.faces, boundaries={1: C.circle(), 2: top},
                   vertex_boundaries={1: tube.rims["bottom"], 2: tube.rims["top"]},
                   parameters={"ht": 0.6})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.relax(levels=2)
        ev.parameters["ht"] = 1.0
        steps = list(recipes.arclength(ev, "ht", 0.05, 12, topology=True))
    fell = [s for s in steps if s.fell]
    assert len(fell) == 1 and fell[0].value > 1.3255
    assert [e.kind for e in fell[0].changes] == ["neck"]
    assert fell[0].energy == pytest.approx(2*np.pi, rel=3e-3)
    assert steps[-1].value > fell[0].value          # and on, the wires apart, the disks flat
    x, ht = ev.vertices, ev.parameters["ht"]
    assert ht == pytest.approx(steps[-1].value)
    assert np.all(np.minimum(np.abs(x[:, 2]), np.abs(x[:, 2] - ht)) < 1e-6)


def test_pinch_off_gives_the_body_number_to_the_larger_piece_either_way():
    # from the sphere bridge's first unstable point, pushed down its mode
    # either way until the neck comes: after the cut body 1 is the larger
    # piece, each piece's volconst its sphere's, and the targets add up to the
    # bridge's (each smaller piece also getting back what collapsing the ring
    # took from its side); the drops relaxed, caps on their spheres
    ev = _sphere_bridge(0.083)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step in recipes.arclength(ev, ev.body(1), -0.003, 12):
            if step.result.negative_modes:
                break
    unstable = ev.save()
    wall = C.sphere((0, 0, -1.3), 1.0, contact_angle=40.0, wet_poles=("north",)).volconst
    h = float(np.median(ev.mesh().edge_lengths()))
    smaller = []
    for sign in (1, -1):
        ev.restore(unstable)
        target = ev.body(1).target
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            way = None
            for _ in range(40):
                u = ev.modes(1).vectors[0]      # (signed by its largest part: kept the same way)
                way = sign*u if way is None else u*np.sign((u*way).sum())
                ev.vertices = ev.vertices + 0.3*h*way/np.linalg.norm(way, axis=1).max()
                ev.command("g 5")
                necks = [e for e in ev.events() if e.kind == "neck"]
                if necks:
                    break
            assert necks
            assert ev.pinch_off(necks[0]) == [2]
        b = ev.bodies()
        assert list(b.ids) == [1, 2] and b.volume[0] > b.volume[1]
        assert b.target.sum() == pytest.approx(target, rel=1e-13)
        assert [ev.body(i).volconst for i in (1, 2)] == pytest.approx([wall, wall])
        assert len(_pieces(ev)) == 2
        smaller.append(ev.vertices[np.asarray(ev.mesh().facets)[
            (np.asarray(ev.mesh().facet_bodies) == 2).any(axis=1)].ravel(), 2].mean())
        assert ev.residual() < 1e-8                 # relaxed (and to the targets:)
        assert b.volume == pytest.approx(b.target, rel=1e-8)
        for i in (1, 2):
            assert ev.body(i).pressure == pytest.approx(
                _cap_on_sphere(ev.body(i).volume, 40.0), rel=0.01)
    assert smaller[0]*smaller[1] < 0          # one way the small drop is on top, the other below


def test_pinch_off_needs_a_neck_and_the_linear_model():
    ev = _two_spheres(0.5)
    with pytest.raises(ValueError, match="no neck"):
        ev.pinch_off()
    ev.command("quadratic")
    with pytest.raises(ValueError, match="linear model"):
        ev.pinch_off()


# ---- stages D and E: merges and contacts (Evolver.merge, Evolver.contact) -----------

def _cap_pressure(volume, theta):
    """The pressure of a cap of this volume on a plane at contact angle theta."""
    t = np.radians(theta)
    R = (volume/(np.pi*(2 - 3*np.cos(t) + np.cos(t)**3)/3))**(1/3)
    return 2/R


def test_two_free_drops_merge_into_one_sphere():
    # one drop grows until it is about to touch the other: the run glues them
    # (Evolver.merge), the merged drop (body 1, the one the run steps) relaxes
    # to the sphere holding both volumes, and the run goes on growing it
    ev = _two_spheres(0.3)
    ev.refine(1)
    for i in (1, 2):
        ev.body(i).target = ev.body(i).volume
    other = ev.body(2).volume
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steps = list(recipes.arclength(ev, ev.body(1), 0.3, 16, topology=True))
    # (one change; a step that didn't settle after it goes on from a relaxed
    # state there, a Step with fell set and no changes)
    k = [i for i, s in enumerate(steps) if s.changes]
    assert len(k) == 1
    fell, before = steps[k[0]], steps[k[0] - 1]
    assert fell.fell and [e.kind for e in fell.changes] == ["touch"]
    assert len(fell.volumes) == 1 and list(ev.bodies().ids) == [1]
    assert fell.volumes[0] == pytest.approx(before.value + other, rel=1e-8)
    R = (3*fell.volumes[0]/(4*np.pi))**(1/3)
    assert fell.energy == pytest.approx(4*np.pi*R*R, rel=0.003)
    assert fell.pressures[0] == pytest.approx(2/R, rel=0.003)
    assert steps[-1].value > fell.value


def test_drops_on_a_plane_merge_and_their_contact_lines_join_into_one_cap():
    # two 120-degree drops on a plane, one growing: they touch above the
    # plane (a merge), the neck sinks, and the two contact lines meet on
    # either side of the dry ground between the drops (contact lines joined,
    # roofed over there, or the neck wetting the plane there), leaving one cap
    ev = _two_drops(120.0, 0.3)
    for i in (1, 2):
        ev.body(i).target = ev.body(i).volume
    other = ev.body(2).volume
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.relax()
        steps = list(recipes.arclength(ev, ev.body(1), 0.2, 12, topology=True))
    fell = [s for s in steps if s.fell]
    assert len(fell) == 1
    kinds = [e.kind for e in fell[0].changes]
    assert kinds[0] == "touch" and ("contact lines" in kinds or "solid" in kinds)
    k = steps.index(fell[0])
    assert fell[0].volumes == pytest.approx([steps[k - 1].value + other], rel=1e-8)
    assert _loops(_border(ev)) == 1 and len(_pieces(ev)) == 1
    for s in steps[k:]:
        assert s.pressures[0] == pytest.approx(_cap_pressure(s.volumes[0], 120.0), rel=0.005)


def test_contact_lines_of_two_drops_meet_and_join():
    # 60-degree drops: their surfaces don't touch before their contact lines
    # meet; the lines are joined (their facing edges glued) and the drops are
    # one cap
    ev = _two_drops(60.0, 0.3)
    for i in (1, 2):
        ev.body(i).target = ev.body(i).volume
    other = ev.body(2).volume
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.relax()
        steps = list(recipes.arclength(ev, ev.body(1), 0.1, 10, topology=True))
    fell = [s for s in steps if s.fell]
    # (the joined surface may come to rest on the plane between them: wetted,
    # in that fall or a later one: which, the last bits of the arithmetic
    # decide, as they differ with what ran before in the process)
    kinds = [(e.kind, e.constraint) for s in fell for e in s.changes]
    assert kinds[0][0] == "contact lines" and fell[0].changes[0].kind == "contact lines"
    assert all(k == ("solid", 1) for k in kinds[1:])
    k = steps.index(fell[0])
    assert fell[0].volumes == pytest.approx([steps[k - 1].value + other], rel=1e-8)
    assert _loops(_border(ev)) == 1
    for s in steps[steps.index(fell[-1]):]:
        assert s.pressures[0] == pytest.approx(_cap_pressure(s.volumes[0], 60.0), rel=0.003)


def test_a_drop_growing_onto_a_sphere_wets_it_into_a_bridge():
    # stage A's drop under a sphere, the sphere at 60 degrees: grown until it
    # is about to touch it, the run puts the surface there on the sphere (a
    # wetted patch round its south pole, whose share of the wetting energy
    # Evolver.energy keeps), and it is a bridge between the plane (at 90
    # degrees: half the bridge between the sphere and its mirror image) and
    # the sphere. Relaxed at volume 2.5 against the exact bridge
    # (docs/stress_helpers.py: pressure 1.396298, energy 5.812735, contact
    # radius 0.69503)
    drop = shapes.cap(0.7, 0.7, nr=6, nt=24)
    ev = pyse.Evolver()
    ev.load_arrays(drop.vertices, drop.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                                2: C.sphere((0, 0, 2.0), 1.0, contact_angle=60.0)},
                   vertex_constraints={1: drop.rims["rim"]}, bodies=[pyse.Body(volume=None)])
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
        assert step.volumes[0] == pytest.approx(step.value, rel=1e-8)
        ev.body(1).target = 2.5
        assert ev.relax().converged
    assert ev.body(1).pressure == pytest.approx(1.396298, rel=0.003)
    assert ev.energy == pytest.approx(5.812735, rel=0.005)
    x = ev.vertices[ev.on_constraint(2)]
    assert np.hypot(x[:, 0], x[:, 1]).mean() == pytest.approx(0.69503, rel=0.01)


def test_a_drop_drained_onto_a_sphere_in_it_dries_a_patch_there():
    # a solid sphere inside a drop, off the axis (its poles clear of the
    # patch: the sphere's integrals take the azimuth round its axis): the drop
    # drains until the liquid over the sphere's side thins away, and a dry
    # patch opens there (a contact line on the sphere, the liquid round it);
    # the volume is the run's, and it settles in a stable state
    drop = shapes.cap(3.0, 3.0, nr=8, nt=32)
    ev = pyse.Evolver()
    ev.load_arrays(drop.vertices, drop.faces,
                   constraints={1: C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                                2: C.sphere((1.3, 0, 1.0), 0.8, contact_angle=30.0,
                                            wet_poles=("north", "south"))},
                   vertex_constraints={1: drop.rims["rim"]}, bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    ev.refine(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.relax()
        steps = []
        for step in recipes.arclength(ev, ev.body(1), -1.0, 40, topology=True):
            steps.append(step)
            if step.fell:
                break
    fell, before = steps[-1], steps[-2]
    assert fell.fell and [e.kind for e in fell.changes] == ["solid"]
    assert fell.changes[0].between == "body" and fell.changes[0].constraint == 2
    assert fell.volumes[0] == pytest.approx(before.value, rel=1e-8)
    assert fell.result.stable and fell.result.health.ok
    edges = np.asarray(ev.mesh().edges)
    assert _loops(edges[ev.on_constraint(2, "edge")]) == 1
    # drying lowered the energy, but not by a pole's share (2 pi 0.64 cos 30)
    assert before.energy - 1.0 < fell.energy < before.energy


def test_merge_and_contact_need_their_events():
    ev = _two_spheres(0.5)
    with pytest.raises(ValueError, match="touch"):
        ev.merge()
    with pytest.raises(ValueError, match="solid"):
        ev.contact()


def test_remesh_protect_with_deletions():
    # the protected edges are carried by id across the deletion that renumbers
    # the rows (the mask was aligned with the rows before it)
    ev = _two_spheres(0.5)
    m = ev.mesh()
    short = m.edge_lengths() < np.median(m.edge_lengths())
    counts = ev.remesh(min_edge=0.3, max_edge=0.6, protect=short)
    assert counts["deleted"] > 0 and counts["split"] >= 0


# ---- the topology check -------------------------------------------------------------

def _drop_on_ball():
    # a 60-degree drop on a unit sphere (a wall from constraints)
    cap = shapes.cap(0.5, 0.25, nr=5, nt=18)
    v = cap.vertices + [0, 0, 1.0]
    rim = np.asarray(cap.rims["rim"])
    v[rim] /= np.linalg.norm(v[rim], axis=1)[:, None]
    ev = pyse.Evolver()
    ev.load_arrays(v, cap.faces, constraints={"ball": C.sphere((0, 0, 0), 1.0, contact_angle=60.0)},
                   vertex_constraints={"ball": rim}, bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    ev.relax()
    return ev


def _found(ev):
    from pysurfaceevolver._surgery import check_topology
    return {p[1].split(" (")[0] for p in check_topology(ev)}


def test_the_topology_check_passes_sound_surfaces():
    assert _found(_drop_on_ball()) == set()
    assert _found(pyse.Evolver(pyse.examples.path("cube.fe"))) == set()


def test_the_topology_check_finds_contact_line_problems():
    ev = _drop_on_ball()
    m = ev.mesh()
    border = int(m.edge_ids[np.flatnonzero(m.edge_valence() == 1)[0]])
    ev.command(f"reverse_orientation edge[{border}]")
    assert _found(ev) == {"a contact-line edge stored against its facet"}
    ev = _drop_on_ball()
    ev.command("dissolve facet[5]")            # (a hole)
    assert "a border edge on no constraint" in _found(ev)


def test_the_topology_check_finds_solids_pinches_and_crossings():
    ev = _drop_on_ball()
    x = ev.vertices.copy()
    top = int(np.argmax(x[:, 2]))
    x[top, 2] -= 0.6                            # through the ball
    ev.vertices = x
    assert {"a vertex inside a solid", "an edge through a solid"} <= _found(ev)
    # two tetrahedra sharing a vertex
    tv = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    tf = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[tv, -tv[1:]], np.r_[tf, np.where(tf == 0, 0, tf + 3)[:, ::-1]],
                   bodies=[pyse.Body(faces=range(4)), pyse.Body(faces=range(4, 8))])
    assert _found(ev) == {"a pinched vertex"}
    # two boxes overlapping
    box = shapes.box(n=4)
    v, f = np.asarray(box.vertices), np.asarray(box.faces)
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[v, v + [0.51, 0.33, 0.27]], np.r_[f, f + len(v)],
                   bodies=[pyse.Body(faces=range(len(f))), pyse.Body(faces=range(len(f), 2*len(f)))])
    assert _found(ev) == {"an edge through a facet"}


def test_the_topology_check_stops_a_run_at_a_broken_change(monkeypatch):
    # PYSE_CHECK_TOPOLOGY=1: a step that leaves a structural problem raises,
    # naming the step
    from pysurfaceevolver import _surgery
    ev = _drop_on_ball()
    monkeypatch.setenv("PYSE_CHECK_TOPOLOGY", "1")
    _surgery._checked(ev, "nothing")
    m = ev.mesh()
    ev.command(f"reverse_orientation edge[{int(m.edge_ids[np.flatnonzero(m.edge_valence() == 1)[0]])}]")
    with pytest.raises(_surgery.TopologyCheckError, match="after a test edit: a contact-line edge"):
        _surgery._checked(ev, "a test edit")


def test_the_nearest_change_is_made_where_a_descent_would_cross_itself():
    # (_surgery._change_nearest, for a descent held back by crossings) two
    # drops' contact lines 0.05 apart: joined into one; 0.6 apart: none
    from pysurfaceevolver._surgery import _change_nearest, check_topology
    ev = _two_drops(60.0, 0.05)
    assert _loops(_border(ev)) == 2
    assert _change_nearest(ev, 2.0, None).kind == "contact lines"
    assert _loops(_border(ev)) == 1 and check_topology(ev) == []
    ev = _two_drops(60.0, 0.6)
    assert _change_nearest(ev, 2.0, None) is None and _loops(_border(ev)) == 2


def test_a_vertex_held_where_two_walls_meet_makes_no_solid_event():
    # a vertex on two walls lies on a third to round-off, 1e-12 through it (a
    # box's corner): no event "gone through" it (no contact can take it); the
    # film's other vertices coming near the third wall still make theirs
    v = np.array([[0, 0, -1e-12], [0.6, 0, 0.3], [0, 0.6, 0.3], [0.5, 0.5, 0.6]])
    ev = pyse.Evolver()
    ev.load_arrays(v, [[0, 1, 3], [0, 3, 2]],
                   constraints={"xw": C.plane((1, 0, 0), 0.0, contact_angle=90.0),
                                "yw": C.plane((0, 1, 0), 0.0, contact_angle=90.0),
                                "zw": C.plane((0, 0, 1), 0.0, contact_angle=90.0)},
                   vertex_constraints={"xw": [0, 2], "yw": [0, 1]})
    found = [e for e in ev.events(2.0) if e.constraint == "zw"]
    assert found and all(e.gap > 0.1 and 0 not in e.vertices for e in found)


def test_a_vertex_through_a_solid_is_left_off_its_guard():
    # a cap on a plane, its top 0.15 into a ball above: the vertices through
    # the ball are a contact to make, not put onto the guard (Evolver would
    # project them out, onto two guards at once far off); those just outside
    # it are guarded, and nothing moves
    from pysurfaceevolver import _surgery as S
    cap = shapes.cap(0.5, 0.25, nr=6, nt=24)
    rim = np.asarray(cap.rims["rim"])
    ev = pyse.Evolver()
    ev.load_arrays(cap.vertices, cap.faces,
                   constraints={"plane": C.plane((0, 0, 1), 0.0, contact_angle=60.0),
                                "ball": C.sphere((0, 0, 1.1), 1.0, contact_angle=30.0)},
                   vertex_constraints={"plane": rim}, bodies=[pyse.Body()])
    x = ev.vertices.copy()
    S.guard(ev)
    assert np.array_equal(ev.vertices, x)
    nums, _, dist, names = ev._constraint_gaps(guards=True)
    names = [str(n) for n in names]
    guarded = np.isnan(dist[:, names.index(f"{S.GUARD_PREFIX}ball")])
    depth = 1.0 - np.linalg.norm(x - [0, 0, 1.1], axis=1)
    h = S._median_edge(ev.mesh())
    assert guarded.any() and (depth[guarded] <= 0.1*h).all() and (depth > 0.1*h).any()


def test_a_closed_piece_enclosing_nothing_is_taken_away():
    # a cube's body, and apart from it a closed piece of four facets lying
    # flat (a tetrahedron squashed into its base plane): no gas in it, it goes
    from pysurfaceevolver import _surgery as S
    box = shapes.box(n=2)
    v, f = np.asarray(box.vertices, float), np.asarray(box.faces)
    flat = np.array([[3, 0, 0], [4, 0, 0], [3.5, 0.8, 0], [3.5, 0.3, 0]], float)
    tet = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]) + len(v)
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[v, flat], np.r_[f, tet], bodies=[pyse.Body(faces=range(len(f)))])
    volume = ev.body(1).volume
    assert S._drop_empty(ev) == 1
    assert ev.counts["facets"] == len(f) and ev.counts["vertices"] == len(v)
    assert ev.body(1).volume == pytest.approx(volume)
    assert S._drop_empty(ev) == 0


def test_a_lone_facet_with_no_area_is_taken_away():
    # a cube's body, and apart from it one facet, its corners in a row (as
    # patches made where a box's walls meet left them along the edge, all
    # three edges border: no edit could take them, and the descent went on
    # down from a relaxed state 400 times for its 0-degree angle); one with
    # area stays
    from pysurfaceevolver import _surgery as S
    box = shapes.box(n=2)
    v, f = np.asarray(box.vertices, float), np.asarray(box.faces)
    for corners, gone in (([[3, 0, 0], [3.02, 0, 0], [3.05, 0, 0]], 1),
                          ([[3, 0, 0], [3.5, 0, 0], [3.2, 0.4, 0]], 0)):
        ev = pyse.Evolver()
        ev.load_arrays(np.r_[v, corners], np.r_[f, [[len(v), len(v) + 1, len(v) + 2]]],
                       bodies=[pyse.Body(faces=range(len(f)))])
        volume = ev.body(1).volume
        assert S._drop_empty(ev) == gone
        assert ev.counts["facets"] == len(f) + 1 - gone
        assert ev.body(1).volume == pytest.approx(volume)


def _hourglass(waist):
    # two triangular double cones joined at a waist `waist` across (a closed
    # surface of 18 facets, one body)
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
    return v, np.array(f)


def _held(v, f):
    ev = pyse.Evolver()
    ev.load_arrays(v, f, bodies=[pyse.Body(faces=range(len(f)))])
    volume = ev.body(1).volume
    ev.body(1).target = volume
    return ev, volume


def test_a_neck_collapsed_below_the_mesh_is_cut():
    # a body pinched to a waist 1e-5 across, three edges round, below what
    # events() sees: cut there, two bodies holding the volume between them
    from pysurfaceevolver import _surgery as S
    ev, volume = _held(*_hourglass(1e-5))
    assert volume > 0
    assert len(S._collapsed_rings(ev.mesh(), S._median_edge(ev.mesh()))) == 1
    assert S._pinch_collapsed(ev) == 1
    bodies = list(ev.bodies().ids)
    assert len(bodies) == 2
    assert sum(ev.body(b).target for b in bodies) == pytest.approx(volume, rel=1e-6)
    assert all(ev.body(b).target > 0.4*volume for b in bodies)
    assert not S._collapsed_rings(ev.mesh(), S._median_edge(ev.mesh()))
    ev.command("g 1")


def test_a_spike_collapsed_below_the_mesh_is_taken_off():
    # the hourglass's top cone replaced by a spike on its waist (three facets
    # to one apex, a filament thinned to a cone 1e-5 across at its base): cut
    # as a neck, its facets would be one point's (Evolver refused every
    # edge, "3 left"); taken off, the surface closed at the waist, the
    # volume as it was
    from pysurfaceevolver import _surgery as S
    v, f = _hourglass(1e-5)
    v = np.r_[v[:7], [[0, 0, 1.5]]]
    f = [row for row in f.tolist() if max(row) <= 6] + [[7, 4 + i, 4 + (i + 1) % 3]
                                                        for i in range(3)]
    ev, volume = _held(v, np.array(f))
    assert len(S._collapsed_rings(ev.mesh(), S._median_edge(ev.mesh()))) == 1
    assert S._pinch_collapsed(ev) == 1
    assert list(ev.bodies().ids) == [1] and ev.counts["vertices"] == 5
    assert ev.body(1).volume == pytest.approx(volume, rel=1e-5)     # (the waist's collapse)
    assert S.check_topology(ev) == [] and not S._collapsed_rings(ev.mesh(), S._median_edge(ev.mesh()))
    ev.command("g 1")


def test_a_spike_with_a_tip_of_two_vertices_is_taken_off():
    # the spike's tip two vertices (each joined only to the waist and the
    # other): cut as a neck, Evolver deleted no edge between them, "2 left";
    # taken off as the one-vertex spike is
    from pysurfaceevolver import _surgery as S
    v, f = _hourglass(1e-5)
    v = np.r_[v[:7], [[0.03, 0, 1.1], [-0.03, 0.02, 1.15]]]
    f = [row for row in f.tolist() if max(row) <= 6] + [[7, 4, 5], [7, 5, 8], [8, 5, 6],
                                                        [8, 6, 4], [7, 8, 4]]
    ev, volume = _held(v, np.array(f))
    assert len(S._collapsed_rings(ev.mesh(), S._median_edge(ev.mesh()))) == 1
    assert S._pinch_collapsed(ev) == 1
    assert list(ev.bodies().ids) == [1] and ev.counts["vertices"] == 5
    assert S.check_topology(ev) == [] and not S._collapsed_rings(ev.mesh(), S._median_edge(ev.mesh()))
    ev.command("g 1")


def test_a_cut_leaves_no_body_without_facets():
    # a cut whose piece is a crumb the seam's tidying takes all of (left
    # without facets, its volume fixed: Evolver's constraint projection
    # singular, ERROR 3005): the piece dissolved, its target back to the
    # body cut, which keeps all of it
    from pysurfaceevolver import _surgery as S
    ev, volume = _held(*_hourglass(1e-5))
    ev.body(1).target = volume + 1e-3
    ev.command("pyse_b := new_body; set body[pyse_b] target 1e-9")     # (a piece with no facets)
    assert S._drop_bare(ev, 1, [2], 0.1) == []
    assert list(ev.bodies().ids) == [1]
    assert ev.body(1).target == pytest.approx(volume + 1e-3 + 1e-9, rel=1e-12)
    ev.command("g 1")


def test_three_tubes_collapsed_at_one_point_are_three_pieces():
    # the facets round a pack's crumpled neck, four vertices 4e-4 to 4e-3
    # apart (two collapsed rings sharing two, a chord across): three
    # filaments meet there, each its own piece (two were looked for, and
    # each ring cut alone left a vertex beside the point, "2 left", or two
    # edges joining the same vertices)
    from types import SimpleNamespace
    from pysurfaceevolver import _surgery as S
    f = np.array([[4, 5, 2], [1, 5, 6], [2, 6, 4], [6, 2, 1], [7, 0, 8], [0, 3, 5],
                  [3, 0, 11], [0, 2, 11], [10, 3, 12], [1, 2, 7], [5, 1, 0], [10, 11, 2],
                  [3, 11, 12], [2, 5, 3], [2, 0, 7], [1, 8, 0], [8, 1, 9], [2, 3, 10],
                  [7, 9, 1]])
    ring = np.arange(4)
    side = S._ring_sides(f, ring)
    assert side is not None and sorted(set(side[4:])) == [1.0, 2.0, 3.0]
    assert [len(set(side[[a, b, c]])) for a, b, c in ((4, 5, 6), (7, 8, 9), (10, 11, 12))] == [1]*3
    lengths = {(0, 1): 3e-4, (1, 2): 5e-4, (0, 2): 7e-4, (0, 3): 3.7e-3, (2, 3): 3.8e-3}
    edges = np.array(sorted({tuple(sorted((int(a), int(b)))) for r in f for a, b in
                             zip(r, np.roll(r, -1))}))
    m = SimpleNamespace(edges=edges, facets=f,
                        edge_lengths=lambda: np.array([lengths.get(tuple(e), 0.1) for e in edges]))
    assert [r.tolist() for r in S._collapsed_clusters(m, 0.16)] == [[0, 1, 2, 3]]


def test_a_collapse_keeps_the_surface_a_manifold():
    # the hourglass's waist edges: merging an edge's ends would join them to
    # the waist's third vertex by two edges (the surface pinched along an
    # edge): the checked collapse refuses (the strict link condition)
    from pysurfaceevolver._edits import REASONS, edit
    ev, _ = _held(*_hourglass(1e-5))
    m = ev.mesh()
    waist = np.flatnonzero(m.edge_lengths() < 1e-3)
    assert len(waist) == 3
    r = edit(ev, "collapse", np.asarray(m.edge_ids)[waist])
    assert [REASONS[abs(int(k))] for k in r] == ["the link condition"]*3
    assert ev.counts["edges"] == len(m.edges)


def test_a_neck_wider_than_the_mesh_is_left():
    from pysurfaceevolver import _surgery as S
    ev, _ = _held(*_hourglass(0.3))
    assert S._pinch_collapsed(ev) == 0
    assert list(ev.bodies().ids) == [1]


def test_rebody_keeps_a_body_whose_surface_is_in_pieces():
    # one body bounded by two separate cubes (as a liquid's pieces joined
    # through the walls): the bodies' rebuild after a change keeps it one,
    # its volume and target; a cut's split makes two
    from pysurfaceevolver import _surgery as S
    box = shapes.box(n=2)
    v, f = np.asarray(box.vertices, float), np.asarray(box.faces)
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[v, v + 3], np.r_[f, f + len(v)], bodies=[pyse.Body(faces=range(2*len(f)))])
    volume = ev.body(1).volume
    ev.body(1).target = volume
    S._rebody(ev)
    assert list(ev.bodies().ids) == [1]
    assert ev.body(1).volume == pytest.approx(volume)
    assert ev.body(1).target == pytest.approx(volume)
    S._rebody(ev, split=True)
    assert len(ev.bodies().ids) == 2


def test_a_line_sweeping_far_from_a_pole_is_no_crossing():
    # contact-line loops on a unit sphere, its pole at +z. A loop round the
    # south grown in a run to bound more than half the sphere (its solid
    # angle, fanned from the pole's antipode, up by over 2 pi), its vertices
    # moving less than their distance from the pole: no crossing. A small
    # loop beside the pole carried round it, the pole now inside: crossed
    from types import SimpleNamespace
    from pysurfaceevolver import _surgery as S

    def loop(centre, r, n=48):
        c = np.asarray(centre, float)/np.linalg.norm(centre)
        u = np.cross(c, [1.0, 0.0, 0.0] if abs(c[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(c, u)
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        return np.cos(r)*c + np.sin(r)*(np.outer(np.cos(t), u) + np.outer(np.sin(t), w))

    def lines(pts):
        pole = np.array([0.0, 0.0, 1.0])
        b = np.roll(pts, -1, axis=0)
        return (None, None, None, None, 1.0, None, S._chord_reach(pts, b, pole),
                float(S._fans(pts, b, pole).sum()), None, None, None, np.ones(len(pts), bool))

    class Surface:              # (the vertices as a mesh, by id)
        def __init__(self, x):
            self.vertices = x

        def mesh(self):
            return SimpleNamespace(vertices=self.vertices, vertex_ids=np.arange(len(self.vertices)))

    def run(start, end):
        ev = Surface(start)
        before = S._line_state(ev, lines(start))
        ev.vertices = end
        return abs(lines(end)[7] - before[0]), S._crossed(ev, before, lines(end))

    jump, crossed = run(loop([0, 0, -1], 0.6), loop([0, 0, -1], np.pi - 1.0))
    assert jump > 2*np.pi and not crossed
    jump, crossed = run(loop([0.6, 0, 1], 0.3), loop([0, 0, 1], 0.3))
    assert jump > 2*np.pi and crossed


def test_a_crumb_and_the_body_it_bounds_are_taken_away():
    # a cube's body and, apart from it, a tetrahedron 0.01 across, its own
    # body: too small for the mesh, it goes with its body, and Evolver steps
    from pysurfaceevolver import _surgery as S
    box = shapes.box(n=2)
    v, f = np.asarray(box.vertices, float), np.asarray(box.faces)
    tip = 0.01*np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) + 3
    tet = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]) + len(v)
    ev = pyse.Evolver()
    ev.load_arrays(np.r_[v, tip], np.r_[f, tet],
                   bodies=[pyse.Body(faces=range(len(f))), pyse.Body(faces=range(len(f), len(f) + 4))])
    assert ev.body(2).volume > 0
    volume = ev.body(1).volume
    assert S._drop_empty(ev) == 1
    assert list(ev.bodies().ids) == [1]
    assert ev.counts["facets"] == len(f)
    assert ev.body(1).volume == pytest.approx(volume)
    ev.command("g 1")


def test_a_run_taking_the_surface_into_a_solid_is_implausible():
    # (a flux-gauged sphere's wetted area and the volume are wrong by the
    # solid's with the surface inside it, and a pack's descent ran away so)
    from pysurfaceevolver import _surgery
    ev = _drop_under_a_sphere()
    before = (ev.energy, _surgery._crossings(ev), ev._target_gap()[0], _surgery._outside(ev))
    assert before[3] == 0
    top = int(np.argmax(ev.vertices[:, 2]))
    ev.command(f"set vertex[{int(ev.mesh().vertex_ids[top])}] z 1.3")
    assert _surgery._outside(ev) == 1
    ev.body(1).target = ev.body(1).volume       # (the volume not what's judged)
    assert _surgery._implausible(ev, before) == "through a wall"


def test_settling_that_fails_leaves_the_surface_as_the_cut_left_it():
    from pysurfaceevolver import _surgery
    ev = _drop_under_a_sphere()
    x, target = ev.vertices.copy(), ev.body(1).target

    class Failed:
        converged, stable = False, None

    def astray(*args, **kwargs):
        ev.command("g 5")
        return Failed()
    ev.command("set vertex z z*1.2 where not fixed and on_constraint 1 == 0")
    x = ev.vertices.copy()
    ev.relax = astray
    ev.newton = astray
    assert _surgery._settle(ev) is None
    assert np.allclose(ev.vertices, x) and ev.body(1).target == target


def test_settling_makes_no_changes_of_topology_of_its_own():
    # (after a cut, as in a descent: relax()'s pops and edge collapses after
    # a pack's sheet had its footprint on a side wall dried left a vertex
    # there pinched, which no contact could separate after)
    from pysurfaceevolver import _surgery
    ev = _drop_under_a_sphere()
    held = []
    relax = ev.relax

    def watched(*args, **kwargs):
        held.append(ev._topology_held)
        return relax(*args, **kwargs)
    ev.relax = watched
    _surgery._settle(ev)
    assert held and all(held) and not ev._topology_held


def test_where_a_run_crossed_the_change_there_is_looked_for_further(monkeypatch):
    # two spheres 0.3 apart on edges of 0.44 (0.68 edges: no event within
    # half an edge); where the last run went through itself there, the
    # touch there within twice that is the change tried (a pack's tongues
    # of contact line 0.55 edges apart crossed in every run, no event
    # there); elsewhere, nothing
    from pysurfaceevolver import _surgery
    ev = _two_spheres(0.3)
    assert not ev.events(0.5)
    tried = []
    monkeypatch.setattr(_surgery, "_try_change", lambda ev, e, *args: tried.append(e) or "")
    _surgery._CROSSED_AT[:] = [np.array([[3.0, 0.0, 0.0]])]
    assert _surgery._change_nearest(ev, 0.5, None) is None and not tried
    _surgery._CROSSED_AT[:] = [np.zeros((1, 3))]
    made = _surgery._change_nearest(ev, 0.5, None)
    assert made is not None and made.kind == "touch" and tried == [made]


def test_the_relaxing_of_a_tidied_state_is_judged_as_runs_are():
    # (a descent's relax() after tidying a relaxed state: one that went
    # wrong, a pack's surface taken deep into a bead, undone, an unconverged
    # result; one that went right kept)
    from pysurfaceevolver import _surgery
    ev = _drop_under_a_sphere()
    x = ev.vertices.copy()
    relax = ev.relax

    class Failed:
        converged, stable, energy = False, None, None

    def astray(*args, **kwargs):
        ev.command("set vertex z z*1.5 where not fixed and on_constraint 1 == 0")
        return Failed()
    ev.relax = astray
    r = _surgery._checked_relax(ev, 1e-8)
    assert not r.converged and np.allclose(ev.vertices, x)
    ev.relax = relax
    assert _surgery._checked_relax(ev, 1e-8).converged


def test_a_run_evolver_fails_in_is_undone_and_implausible():
    # (a pack's relax() run met an internal error, cone_analyze on a mesh a
    # failed change left; the descent goes on as from a run gone wrong)
    from pysurfaceevolver import _surgery
    from pysurfaceevolver._errors import EvolverError
    ev = _drop_under_a_sphere()
    x = ev.vertices.copy()

    def run(n):
        ev.command("g 3")
        raise EvolverError("ERROR 1303: Internal error: cone_analyze vertex 7", 1303)
    result, start, before = _surgery._away_from_poles(ev, run, 10)
    assert not result.converged and np.allclose(ev.vertices, x)
    assert _surgery._implausible(ev, before) == "Evolver failed"
    _surgery._away_from_poles(ev, lambda n: ev.command(f"g {n}"), 2)
    assert _surgery._implausible(ev, before) != "Evolver failed"


def _sheet_on_a_wall(thin=0.03):
    """A thin sheet of liquid standing on the wall x = 0 (90 degrees: a
    mirror), half a flattened hemisphere of radius 0.7, thinnest where it
    meets the wall at mid-height (1.4 thin across: 0.042)."""
    cap = shapes.cap(0.7, 0.7, nr=8, nt=32)
    v = np.asarray(cap.vertices, float)
    x, y, z = v[:, 2], v[:, 0], v[:, 1]
    ev = pyse.Evolver()
    ev.load_arrays(np.c_[x, y*(thin + 0.6*x + 2*z**2), z], cap.faces,
                   constraints={1: C.plane((1, 0, 0), 0.0, contact_angle=90.0)},
                   vertex_constraints={1: cap.rims["rim"]}, bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    return ev


def test_a_sheet_thinnest_at_a_wall_ruptures_there():
    # (mirrored in the wall, a sheet twice as wide, its hole opening in the
    # middle, on the mirror: its footprint cut in two, the liquid one body)
    from pysurfaceevolver import _surgery
    ev = _sheet_on_a_wall()
    v0 = ev.body(1).volume
    neck = [e for e in ev.events() if e.kind == "neck"]
    assert neck and abs(neck[0].where[0]) < 1e-9 and abs(neck[0].where[2]) < 0.05
    assert len(ev.contact_lines(1)) == 1
    ev.pinch_off(neck[0], relax=False)
    lines = ev.contact_lines(1)
    assert len(lines) == 2 and all(np.allclose(line[0], line[-1]) for line in lines)
    assert sorted(np.sign(np.mean(line[:-1, 2])) for line in lines) == [-1, 1]
    assert list(ev.bodies().ids) == [1] and ev.body(1).volume == pytest.approx(v0, rel=1e-9)
    assert _surgery.check_topology(ev) == [] and _surgery._crossings(ev) == 0


def test_a_contact_separates_only_its_own_pinches(monkeypatch):
    # a vertex the contact line already goes through twice away from the
    # event (a pack's, on a side wall, unseparable: every contact on that
    # wall after was refused) is left to itself; one among the event's
    # vertices is the change's
    from pysurfaceevolver import _edits, _surgery
    ev = _sheet_on_a_wall()
    m = ev.mesh()
    row = int(np.flatnonzero(ev.on_constraint(1))[0])
    vid = int(np.asarray(m.vertex_ids)[row])
    monkeypatch.setattr(_surgery, "_pinch_rows", lambda ev, m, num: np.array([row]))
    left = _surgery._pinches_elsewhere(ev, 1, [])
    assert left == {vid} and not _surgery._pinches_elsewhere(ev, 1, [vid])

    def refused(*args, **kwargs):
        raise AssertionError("separated a pinch elsewhere")
    monkeypatch.setattr(_edits, "edit", refused)
    _surgery._separate_pinches(ev, 1, left)


def test_a_sheet_is_an_event_as_two_sheets_are_not_as_a_neck():
    # (a neck counts within NECK times the gap two sheets do, the ring round
    # it under-resolved; a sheet's faces are two sheets: 0.042 across on
    # edges of 0.16, under half an edge, it is a neck event to rupture; 0.14
    # across, not yet: at NECK times, a 16-bead pack's sheets ruptured 1-1.8
    # edges thick, the liquid came back over the cut, again and again)
    from pysurfaceevolver._analysis import EVENT_FACTOR
    thin = [e for e in _sheet_on_a_wall(0.03).events() if e.kind == "neck"]
    assert len(thin) == 1 and thin[0].gap < EVENT_FACTOR*thin[0].scale
    assert not [e for e in _sheet_on_a_wall(0.1).events() if e.kind == "neck"]


def _half_neck(waist=0.06, nz=11, nt=8):
    """Half a liquid bridge between the planes z = 0 and z = 1 lying against
    the wall y = 0 (all at 90 degrees: mirrors), its axis along the wall,
    thinned to a neck `waist` across at mid-height: the neck's ring runs
    from the wall's contact line round to it again."""
    z = np.linspace(0.0, 1.0, nz + 1)
    r = 0.6 - (0.6 - waist/2)*np.sin(np.pi*z)**2
    t = np.linspace(0.0, np.pi, nt + 1)
    v = np.array([[rr*np.cos(tt), rr*np.sin(tt), zz] for zz, rr in zip(z, r) for tt in t])

    def k(i, j):
        return i*(nt + 1) + j
    f = []
    for i in range(nz):
        for j in range(nt):
            a, b, c, d = k(i, j), k(i, j + 1), k(i + 1, j + 1), k(i + 1, j)
            f += [[a, b, c], [a, c, d]] if (i + j) % 2 else [[a, b, d], [b, c, d]]
    ev = pyse.Evolver()
    ev.load_arrays(v, f,
                   constraints={"wall": C.plane((0, 1, 0), 0.0, contact_angle=90.0),
                                "bottom": C.plane((0, 0, 1), 0.0, contact_angle=90.0),
                                "top": C.plane((0, 0, -1), -1.0, contact_angle=90.0)},
                   vertex_constraints={"wall": [k(i, j) for i in range(nz + 1) for j in (0, nt)],
                                       "bottom": [k(0, j) for j in range(nt + 1)],
                                       "top": [k(nz, j) for j in range(nt + 1)]},
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    return ev


@pytest.mark.parametrize("nz", [10, 11])
def test_a_neck_against_a_wall_is_cut_into_two_bodies(nz):
    # (the neck's axis along the wall: its waist's vertices are its flanks
    # beside the wall, and their normals put the axis across it, the ring
    # then beside the wall on both flanks, not over the neck; cut there, the
    # neck stayed, the body one, a vertex pinched (10 rows, the waist on
    # one) or the pieces one body (11))
    from pysurfaceevolver import _surgery
    ev = _half_neck(nz=nz)
    v0 = ev.body(1).volume
    neck = [e for e in ev.events() if e.kind == "neck"]
    assert len(neck) == 1
    ev.pinch_off(neck[0], relax=False)
    assert list(ev.bodies().ids) == [1, 2]
    volumes = [ev.body(i).volume for i in (1, 2)]
    assert sum(volumes) == pytest.approx(v0, rel=0.02) and min(volumes) > 0.45*v0
    assert len(ev.contact_lines("wall")) == 2
    assert _surgery.check_topology(ev) == [] and _surgery._crossings(ev) == 0


def test_a_neck_against_a_wall_whose_ring_wont_collapse_has_its_footprint_dried(monkeypatch):
    # (Evolver's deletion refusing the ring's edges, "the ring didn't
    # collapse to one vertex (3 left)": the footprint under the neck dried
    # across instead, the liquid parted at the wall into two bodies; where
    # that leaves the neck, the refusal stands and the surface is as it was)
    from pysurfaceevolver import _surgery

    def refused(*args, **kwargs):
        raise ValueError("pinch_off(): the ring didn't collapse to one vertex (3 left)")
    monkeypatch.setattr(_surgery, "_pinch_ring", refused)
    ev = _half_neck(nz=11)
    v0 = ev.body(1).volume
    neck = [e for e in ev.events() if e.kind == "neck"]
    ev.pinch_off(neck[0], relax=False)
    assert list(ev.bodies().ids) == [1, 2]
    volumes = [ev.body(i).volume for i in (1, 2)]
    assert sum(volumes) == pytest.approx(v0, rel=0.01) and min(volumes) > 0.45*v0
    assert len(ev.contact_lines("wall")) == 2 and len(_pieces(ev)) == 2
    assert _surgery.check_topology(ev) == [] and _surgery._crossings(ev) == 0
    ev = _half_neck(nz=10)
    x = ev.vertices
    neck = [e for e in ev.events() if e.kind == "neck"]
    with pytest.raises(ValueError, match="didn't collapse"):
        ev.pinch_off(neck[0], relax=False)
    assert np.array_equal(ev.vertices, x) and list(ev.bodies().ids) == [1]


def test_two_fans_joined_along_an_edge_on_a_wall_are_unzipped():
    # (a neck against a wall collapsed to its two ends on the wall, the edge
    # between them kept by Evolver: the checked unzip makes it two
    # contact-line edges, one per fan, and parts the ends, one vertex per
    # fan; the patch check counting each vertex once, the other end, pinched
    # until it is parted too, not twice)
    from pysurfaceevolver._edits import edit
    v = np.array([[0, 0, 0], [0.2, 0, 0], [-0.1, 0, 0.15], [0.1, 0.15, 0.15], [0.3, 0, 0.15],
                  [-0.1, 0, -0.15], [0.1, 0.15, -0.15], [0.3, 0, -0.15]], float)
    f = [[0, 1, 3], [0, 3, 2], [1, 4, 3], [1, 0, 6], [0, 5, 6], [1, 6, 7]]
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints={"wall": C.plane((0, 1, 0), 0.0, contact_angle=90.0)},
                   vertex_constraints={"wall": [0, 1, 2, 4, 5, 7]})

    def lines(m):
        return int(((m.edge_valence() == 1) & ev.on_constraint("wall", "edge")).sum())
    m = ev.mesh()
    assert lines(m) == 4
    joint = [k for k, (a, b) in enumerate(np.asarray(m.edges)) if {int(a), int(b)} == {0, 1}]
    assert list(edit(ev, "unzip", [int(m.edge_ids[joint[0]])])) == [0]
    m = ev.mesh()
    assert len(m.vertices) == 10 and lines(m) == 6
    assert len(_pieces(ev)) == 2


def _corner():
    """A few facets in a box's corner (side walls x = 0 and y = 0, 90
    degrees) by a bead in it, 0.015 from both: vertex 0 free, 0.25 from both
    walls and 0.08 from the bead; vertex 1 on the wall y = 0 (a contact
    line's), 0.1 from the wall x = 0; vertex 2 on the wall x = 0 where the
    bead nearly touches it, 0.015 from the bead."""
    v = np.array([[0.25, 0.25, 1.0], [0.1, 0.0, 1.15], [0.0, 1.015, 1.02], [0.3, 0.0, 0.85],
                  [0.0, 0.3, 0.85], [0.0, 0.8, 1.2]], float)
    f = [[0, 1, 3], [0, 4, 1], [0, 5, 4], [0, 2, 5]]
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints={
        "wallxlo": C.plane((1, 0, 0), 0.0, contact_angle=90.0),
        "wallylo": C.plane((0, 1, 0), 0.0, contact_angle=90.0),
        "bead": C.sphere((1.015, 1.015, 1.0), 1.0, contact_angle=15.0, gauge="flux")},
        vertex_constraints={"wallylo": [1, 3], "wallxlo": [2, 4, 5]})
    return ev


def test_the_guards_keep_vertices_from_every_wall_near_them():
    # a free vertex in the corner, guarded against both side walls only,
    # went into the bead in the corner (a pack's 151 runs undone so in one
    # descent): guarded against all three; a contact line's vertex on one
    # side wall against the other (it went out through it) and the bead,
    # filling its three; not against a wall as good as tangent to its own
    # (the bead where it nearly touches the side wall: Evolver's projection
    # singular, ERROR 3005)
    from pysurfaceevolver import _surgery as S
    from pysurfaceevolver._engine import GUARD_PREFIX, surface
    ev = _corner()
    surface.edge_scale = 0.2
    x = ev.vertices.copy()
    S.guard(ev)
    nums, _, dist, names = ev._constraint_gaps(guards=True)
    names = [str(n) for n in names]

    def guards(v):
        return sorted(n[len(GUARD_PREFIX):] for c, n in enumerate(names)
                      if n.startswith(GUARD_PREFIX) and np.isnan(dist[v, c]))
    assert guards(0) == ["bead", "wallxlo", "wallylo"]
    assert guards(1) == ["bead", "wallxlo"]
    assert guards(2) == []
    assert np.array_equal(ev.vertices, x)


def test_vertices_go_on_and_off_constraints_in_one_call():
    # (guard()'s and _unguard()'s: several constraints, a mask column each)
    from pysurfaceevolver import _core
    ev = _corner()
    nums, _, _, names = ev._constraint_gaps()
    names = [str(n) for n in names]
    cols = [int(nums[names.index(k)]) for k in ("bead", "wallxlo")]
    mask = np.zeros((ev.counts["vertices"], 2), dtype=np.uint8)
    mask[0, 0] = mask[1, 1] = 1
    ev._call(_core.set_vertex_constraints, np.array(cols, dtype=np.int32), mask, True)
    assert list(np.flatnonzero(ev.on_constraint("bead"))) == [0]
    assert list(np.flatnonzero(ev.on_constraint("wallxlo"))) == [1, 2, 4, 5]
    # (projected onto them, as Evolver's set does)
    assert abs(np.linalg.norm(ev.vertices[0] - [1.015, 1.015, 1.0]) - 1.0) < 1e-9
    assert abs(ev.vertices[1, 0]) < 1e-12
    ev._call(_core.set_vertex_constraints, np.array(cols, dtype=np.int32), mask, False)
    assert not ev.on_constraint("bead").any()
    assert list(np.flatnonzero(ev.on_constraint("wallxlo"))) == [2, 4, 5]


def test_unguarding_keeps_the_mesh_and_the_gaps_without_guards():
    # (nothing moves: kept for the new surface version, not taken again)
    from pysurfaceevolver import _engine, _surgery as S
    from pysurfaceevolver._engine import surface
    ev = _corner()
    surface.edge_scale = 0.2
    S.guard(ev)
    m = ev.mesh()
    _, _, dist, _ = ev._constraint_gaps()
    S._unguard(ev)
    assert ev.mesh() is m
    kept = ev._constraint_gaps()[2]
    _engine._gaps_cache = None
    assert np.array_equal(kept, ev._constraint_gaps()[2], equal_nan=True)
    assert np.array_equal(kept, dist, equal_nan=True)
    assert not S._guarded(ev)


def test_a_contact_lines_split_takes_its_ends_guards():
    # (the contact-line edges split as gradient steps go, _core.line_chop: a
    # midpoint without its ends' guards went out through the side wall at
    # the box's edge)
    from pysurfaceevolver import _core, _surgery as S
    from pysurfaceevolver._engine import GUARD_PREFIX, surface
    ev = _corner()
    surface.edge_scale = 0.2
    S.guard(ev)
    n = ev.counts["vertices"]
    was = _core.line_chop(0.1)
    try:
        ev.command("g 1")
    finally:
        _core.line_chop(was)
    assert ev.counts["vertices"] > n
    nums, _, dist, names = ev._constraint_gaps(guards=True)
    names = [str(c) for c in names]
    on_wall = np.isnan(dist[:, names.index("wallylo")])
    guarded = np.isnan(dist[:, names.index(f"{GUARD_PREFIX}wallxlo")])
    new = np.arange(ev.counts["vertices"]) >= n
    assert (new & on_wall).any() and guarded[new & on_wall].all()


def _hole(a, b):
    """A film over the plane z = 0 with a triangular hole `a` across, its rim
    on the plane (a patch's contact line of three edges), a ring of edges
    `b` long round it."""
    t3, t6 = np.arange(3)*2*np.pi/3, np.arange(6)*np.pi/3 + np.pi/6
    v = np.r_[np.c_[a/np.sqrt(3)*np.cos(t3), a/np.sqrt(3)*np.sin(t3), np.zeros(3)],
              np.c_[b*np.cos(t6), b*np.sin(t6), np.full(6, 0.3*b)]]
    f = []
    for i in range(3):
        j = (i + 1) % 3
        f += [[i, 3 + 2*i, j], [j, 3 + 2*i, 3 + 2*i + 1], [j, 3 + 2*i + 1, 3 + (2*i + 2) % 6]]
    ev = pyse.Evolver()
    ev.load_arrays(v, f, constraints={"ground": C.plane((0, 0, 1), 0.0, contact_angle=60.0)},
                   vertex_constraints={"ground": [0, 1, 2]})
    return ev


def test_a_patch_is_closed_as_shrunk_against_the_edges_round_it():
    # (a contact where the mesh was refined, edges 0.05 for h 0.165, opened
    # a patch of a facet, closed as shrunk at the next tidying, and the
    # surface lay on the wall again, its contact refused as reopening)
    from pysurfaceevolver import _surgery as S
    assert S.close_patches(_hole(0.03, 0.05), 0.165) == 0
    assert S.close_patches(_hole(0.003, 0.05), 0.165) == 1
    assert S.close_patches(_hole(0.03, 0.3), 0.165) == 1


def test_a_sheet_thinner_than_its_facets_opens_a_hole():
    # a slab of liquid 0.003 thick on facets of 0.1, its faces' grids offset:
    # the tube between facing facets lies all but flat and folds back on the
    # faces whatever the pair (refused at every pair, and a pack's descent
    # ran away round such a sheet); allowed for a sheet's hole, it opens,
    # nothing through itself, and gradient steps go on from it
    from pysurfaceevolver import _surgery as S
    t = 0.003
    box = shapes.box(size=(1.0, 1.0, t), n=10)
    v = np.asarray(box.vertices, float).copy()
    top = (v[:, 2] > t/2) & (np.abs(v[:, :2] - 0.5) < 0.45).all(axis=1)
    v[top] += [0.035, 0.02, 0.0]
    ev = pyse.Evolver()
    ev.load_arrays(v, box.faces, bodies=[pyse.Body(faces=range(len(box.faces)))])
    m = ev.mesh()
    n = ev.counts["facets"]
    assert S._sheet_hole(ev, m, np.asarray(m.vertices), np.asarray(m.facets),
                         np.array([0.5, 0.5, t/2]), 0.2)
    assert ev.counts["facets"] == n + 4
    assert [p for p in S.check_topology(ev) if p[0] < 8] == [] and S._crossings(ev) == 0


def test_a_necks_waist_is_its_narrowest_point():
    # (a neck's event vertices can be a whole thin film; its waist, whose
    # normals tell a sheet from a tube and set the cut's axis, is those
    # within its gap and two edges of its narrowest point)
    from pysurfaceevolver._events import waist
    x = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [1, 0, 0], [0, 1, 0]], float)
    assert list(waist(x, np.arange(5), [0, 0, 0], 0.01, 0.04)) == [0, 1, 2]
    assert list(waist(x, np.arange(5), [0, 0, 0], 0.01, 0.01)) == [0, 1, 2, 3, 4]


def test_a_footprint_under_a_dome_is_told_from_a_sheets():
    # (a neck lying against a wall domes over its footprint, the liquid under
    # it, dried by laying the dome onto the wall; a sheet's faces rise off the
    # wall with gas over the strip between them, joined only round its edge)
    from pysurfaceevolver import _surgery
    for make, domed in ((_drop_under_a_sphere, True), (_sheet_on_a_wall, False)):
        ev = make()                 # (one at a time: every handle is the one engine)
        m = ev.mesh()
        x = np.asarray(m.vertices)
        on = ev.on_constraint(1)
        rim = np.flatnonzero(on)
        if domed:
            a = rim[np.argmax(x[rim, 0])]
            b = rim[np.argmin(x[rim, 0])]
        else:
            mid = rim[np.abs(x[rim, 2]) < 1e-9]
            a, b = mid[np.argmax(x[mid, 1])], mid[np.argmin(x[mid, 1])]
        h = float(np.median(m.edge_lengths()))
        nums, _, _, names = ev._constraint_gaps()
        num = int(nums[list(names).index(names[0])])
        assert _surgery._domed(ev, m, x, int(a), int(b), num, 1, h) is domed


# ---- the flux gauge (constraints.sphere(gauge="flux")) ---------------------------

def _drop_on_ball_flux():
    # _drop_on_ball's drop on a sphere with no pole: its wetted area the flux
    # of the sphere's point source through the drop
    cap = shapes.cap(0.5, 0.25, nr=5, nt=18)
    v = cap.vertices + [0, 0, 1.0]
    rim = np.asarray(cap.rims["rim"])
    v[rim] /= np.linalg.norm(v[rim], axis=1)[:, None]
    ev = pyse.Evolver()
    ev.load_arrays(v, cap.faces,
                   constraints={"ball": C.sphere((0, 0, 0), 1.0, contact_angle=60.0, gauge="flux")},
                   vertex_constraints={"ball": rim}, bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    ev.relax()
    return ev


def _deepest_chord(ev, wall="ball"):
    """How deep the middle of an edge between two vertices on the unit ball
    (on two facets, not a contact line's) lies inside it."""
    m = ev.mesh()
    on, e = ev.on_constraint(wall), np.asarray(m.edges)
    chord = (m.edge_valence() == 2) & on[e[:, 0]] & on[e[:, 1]]
    if not chord.any():
        return 0.0
    return 1 - float(np.linalg.norm(np.asarray(m.vertices)[e[chord]].mean(axis=1), axis=1).min())


def test_a_chord_through_a_bead_is_lifted_out_of_it():
    # (_surgery._lift_chords) the drop's row next to its contact line put on
    # the ball: the facets between have every corner on it, their edges
    # across it are chords through it, which nothing holds off the ball and
    # the remeshing can't split (their middles are inside it). Split, each
    # middle put out on the ball, free
    from pysurfaceevolver import _surgery
    from pysurfaceevolver._engine import surface
    ev = _drop_on_ball_flux()
    m = ev.mesh()
    rim, edges = ev.on_constraint("ball"), np.asarray(m.edges)
    row = np.unique(edges[rim[edges[:, 0]] != rim[edges[:, 1]]].ravel())
    for v in np.asarray(m.vertex_ids)[row[~rim[row]]]:
        ev.command(f"set vertex[{int(v)}] constraint ball")
    ev.command("recalc")
    surface.edge_scale = 0.01           # (a chord counts from a tenth of an edge deep)
    assert _deepest_chord(ev) > 0.001
    facets = ev.counts["facets"]
    lifted = _surgery._lift_chords(ev)
    assert lifted > 0 and ev.counts["facets"] == facets + 2*lifted
    assert _deepest_chord(ev) <= 0.001
    assert np.linalg.norm(ev.vertices, axis=1).min() > 1 - 1e-9
    assert _surgery._lift_chords(ev) == 0


def test_a_stalled_descent_is_told_by_its_state():
    # (_surgery._descend: runs that leave the counts and the energy as they
    # were go to the remeshing at once; a relaxed state that comes back to
    # itself is taken where only its mesh is wanting)
    from types import SimpleNamespace
    from pysurfaceevolver import _surgery
    state = ((100, 290, 190, 1), -93.119278)
    assert not _surgery._same_state(state, None)
    assert _surgery._same_state(state, ((100, 290, 190, 1), -93.119281))
    assert not _surgery._same_state(state, ((100, 290, 190, 1), -93.1193))
    assert not _surgery._same_state(state, ((101, 290, 190, 1), -93.119278))
    run = ((100, 290, 190, 1), -97.967412, 3.85)
    assert _surgery._stalled_run(run, ((100, 290, 190, 1), -97.9674, 3.88))
    assert not _surgery._stalled_run(run, ((100, 290, 190, 1), -97.9674, 9.0))   # converging
    assert not _surgery._stalled_run(run, ((100, 290, 190, 1), -97.966, 3.88))
    issues = ["degenerate facets (smallest angle 0.14 degrees, 27 below 5)",
              "the surface nearly touches itself (0.037 edge lengths)"]
    assert _surgery._mesh_only(SimpleNamespace(issues=issues))
    assert not _surgery._mesh_only(SimpleNamespace(issues=issues + ["not in equilibrium (residual 2.6)"]))
    assert not _surgery._mesh_only(None)


def test_a_change_where_one_was_made_is_put_off_till_nearer():
    # (_surgery._put_off) within a descent: the opposite of a change made
    # (on its solid, within three of its edges), or the same within an edge
    # (it came back), waits until it is nearer than half that one's gap; a
    # refusal puts off its own kind only; a surface resting on a solid isn't
    # put off
    from pysurfaceevolver import _surgery
    from pysurfaceevolver._events import Event

    def tried(kind, gap, x, sense, what="", solid="bead1"):
        return _surgery._Tried(kind, frozenset([solid]), np.array([x, 0.0, 0.0]), 0.1, gap/0.1,
                               sense, what)
    put_off = _surgery._put_off
    made = [tried("solid", 0.01, 0.0, -1, "made")]                # a dry patch
    assert put_off(tried("solid", 0.01, 0.2, 1), made)            # wetted back
    assert put_off(tried("contact lines", 0.02, 0.1, 1), made)    # its lines joined back
    assert put_off(tried("solid", 0.01, 0.05, -1), made)          # come back
    assert not put_off(tried("solid", 0.01, 0.2, -1), made)       # another beside it
    assert not put_off(tried("solid", 0.004, 0.2, 1), made)       # nearer
    assert not put_off(tried("contact lines", 0.0, 0.1, 1), made)     # met
    assert not put_off(tried("solid", 0.01, 0.4, 1), made)        # elsewhere
    assert not put_off(tried("solid", 0.01, 0.2, 1, solid="bead2"), made)
    assert not put_off(tried("contact lines", 0.02, 0.05, -1), made)     # the same way on
    refused = [tried("solid", 0.01, 0.0, -1, "refused")]
    assert put_off(tried("solid", 0.01, 0.2, -1), refused)
    assert not put_off(tried("contact lines", 0.02, 0.1, 1), refused)
    through = [tried("solid", -0.01, 0.0, -1, "made")]
    assert put_off(tried("solid", -0.015, 0.0, 1), through)
    assert not put_off(tried("solid", -0.025, 0.0, 1), through)
    assert not put_off(tried("solid", 0.01, 0.0, 1), [])
    # (resting on the solid, held there by its guard: it comes no nearer)
    assert not put_off(tried("solid", 0.003, 0.05, -1), made)
    assert not put_off(tried("solid", 0.003, 0.2, 1), made)

    def event(kind, between=None):
        return Event(kind, 0.01, 0.1, np.zeros(3), np.array([0]), "bead1", between)
    assert [_surgery._sense(None, event(k)) for k in ("neck", "touch")] == [-1, 1]
    assert [_surgery._sense(None, event("solid", b)) for b in ("body", "outside")] == [-1, 1]


def test_a_patch_growing_by_resting_contacts_is_no_repeat(monkeypatch):
    # (_surgery._again) made three times at one place, a change is not made
    # there again; but a surface resting on a solid beside its contact line
    # there is the patch growing (the guard holds it at the solid): only its
    # refusals and undoing count
    from types import SimpleNamespace
    from pysurfaceevolver import _surgery
    from pysurfaceevolver._events import Event
    monkeypatch.setattr(_surgery, "_edge_scale", lambda ev: 0.1)
    monkeypatch.setattr(_surgery, "_beside_line", lambda ev, e, h: True)
    at = [("solid", "bead1", np.zeros(3))]

    def event(gap):
        return Event("solid", gap, 0.1, np.zeros(3), np.array([0]), "bead1", "body")
    again = _surgery._again
    assert again(None, event(0.01), at*3, [], [])               # coming, made thrice
    assert not again(None, event(0.001), at*3, [], [])          # resting beside its line
    assert again(None, event(0.001), [], at*3, [])              # refused thrice
    assert again(None, event(0.001), [], [], at)                # undone
    monkeypatch.setattr(_surgery, "_beside_line", lambda ev, e, h: False)
    assert again(None, event(0.001), at*3, [], [])              # resting elsewhere


def test_a_drop_on_a_flux_sphere_is_the_exact_cap():
    # (refined, its pressure 1.3%, 0.32%, 0.08% off: second order)
    ev = _drop_on_ball_flux()
    ev.refine()
    ev.relax()
    V = ev.body(1).volume
    assert V == pytest.approx(ev.body(1).target, rel=1e-8)
    assert ev.body(1).pressure == pytest.approx(_cap_on_sphere(V, 60.0), rel=5e-3)


def _vertex_through_the_ball(ev, n=41, vid=None):
    """The energies and volumes with the drop's top vertex (or vertex vid)
    moved down the z axis through the sphere's centre, to z = -0.5."""
    if vid is None:
        m = ev.mesh()
        vid = int(m.vertex_ids[int(np.argmax(np.asarray(m.vertices)[:, 2]))])
    out = []
    for z in np.linspace(0.5, -0.5, n):
        ev.command(f"set vertex[{vid}] x 0; set vertex[{vid}] y 0; set vertex[{vid}] z {z!r}; recalc")
        out.append((ev.energy, ev.body(1).volume))
    return np.asarray(out)


def test_a_surface_through_a_flux_sphere_has_no_jump_at_its_centre():
    # (inside the sphere, where liquid never is, the field is bounded: a
    # surface pushed across the centre changed the wetted area by 4 pi R^2,
    # Evolver's line search found the jump, and a pack's descent ran away)
    ev = _drop_on_ball_flux()
    E, V = _vertex_through_the_ball(ev).T
    assert np.abs(np.diff(E)).max() < 0.2           # (a jump: 4 pi cos 60 = 6.3)
    assert np.abs(np.diff(V)).max() < 0.05          # (a jump: 4 pi/3 = 4.2)



def test_the_flux_field_is_smooth_through_the_sphere():
    # (capped at the sphere itself, the field's kink sat on the contact lines,
    # where a chord's facets dip into the sphere: their quadrature points
    # either side of it, Newton's quadratic model failed there, and 8 of 18
    # 4-bead packs' descents never settled; the cap is now deep inside)
    ev = _drop_on_ball_flux()
    x = ev.vertices
    top = int(np.argmax(x[:, 2]))
    energy = []
    for z in np.linspace(1.0, 0.85, 151):      # (its facets' points through r = 1)
        y = x.copy()
        y[top] = [0, 0, z]
        ev.vertices = y
        energy.append(ev.energy)
    d2 = np.abs(np.diff(energy, 2))
    assert d2.max() < 5*np.median(d2)               # (59 times, capped at R)

def test_the_flux_gauge_in_c_is_the_datafiles():
    # (the integrands evaluated natively, inside the sphere too, against
    # Evolver's interpreter of the datafile's expressions)
    from pysurfaceevolver import _core
    ev = _drop_on_ball_flux()
    m = ev.mesh()
    vid = int(m.vertex_ids[int(np.argmax(np.asarray(m.vertices)[:, 2]))])
    native = _vertex_through_the_ball(ev, n=9, vid=vid)
    _core.native_enable(False)
    try:
        interpreted = _vertex_through_the_ball(ev, n=9, vid=vid)
    finally:
        _core.native_enable(True)
    assert np.allclose(native, interpreted, rtol=1e-10, atol=1e-10)


def _drop_among_flux_spheres(radii, angle=60.0):
    """_drop_on_ball_flux's drop with more flux spheres below the ball, apart
    from the liquid (radii: theirs, in a row of rows): every facet integrand
    has their point sources, the ball's contact line their monopoles."""
    cap = shapes.cap(0.5, 0.25, nr=4, nt=12)
    v = cap.vertices + [0, 0, 1.0]
    rim = np.asarray(cap.rims["rim"])
    v[rim] /= np.linalg.norm(v[rim], axis=1)[:, None]
    cons = {"ball": C.sphere((0, 0, 0), 1.0, contact_angle=angle, gauge="flux")}
    for i, R in enumerate(radii):
        cons[f"s{i}"] = C.sphere((0.6*(i % 9) - 2.4, 0.6*(i//9) - 2.4, -2.0 - 0.1*(i % 3)), R,
                                 contact_angle=angle, gauge="flux")
    ev = pyse.Evolver()
    ev.load_arrays(v, cap.faces, constraints=cons, vertex_constraints={"ball": rim},
                   bodies=[pyse.Body(volume=None)])
    ev.body(1).target = ev.body(1).volume
    ev.command("convert_to_quantities")
    return ev


@pytest.mark.parametrize("radii", [[0.2]*70, list(np.linspace(0.1, 0.25, 70))])
def test_the_flux_gauge_in_c_is_the_datafiles_with_many_spheres(radii):
    # (71 spheres: past the 64 terms C once took, the rest went to the
    # interpreter; equal spheres' sums are taken once for the volume's and
    # the wetting's integrands, unequal ones' column by column; the contact
    # line's monopoles, with the angle floor raised past the angle too)
    from pysurfaceevolver import _core
    ev = _drop_among_flux_spheres(radii, angle=15.0)
    m = ev.mesh()
    vid = int(m.vertex_ids[int(np.argmax(np.asarray(m.vertices)[:, 2]))])
    saved = ev._save()
    found = []
    for native in (True, False):
        _core.native_enable(native)
        try:
            for floor in (0.0, 17.0):
                ev.restore(saved)
                ev.command(f"pyse_angle_floor := {floor}; recalc")
                out = list(_vertex_through_the_ball(ev, n=5, vid=vid).ravel())
                ev.restore(saved)
                ev.command(f"pyse_angle_floor := {floor}; recalc")
                ev.command("g 2")                # (the gradient: first derivatives)
                out += list(np.asarray(ev.vertices).ravel()) + [ev.energy]
                found.append(out)
        finally:
            _core.native_enable(True)
    assert np.allclose(found[:2], found[2:], rtol=1e-10, atol=1e-10)
    # second derivatives: Newton steps converge as with C's 40 spheres, 0.29
    # to 6.5e-11 in six (the interpreter's second derivatives of these powers
    # are wrong: 2e-6)
    ev.restore(saved)
    ev.command("pyse_angle_floor := 0; recalc")
    ev.command("g 10")
    for _ in range(6):
        ev.command("hessian")
    assert ev.residual() < 1e-9


@pytest.mark.parametrize("radii", [[0.2]*70, list(np.linspace(0.1, 0.25, 70))])
def test_the_flux_gauge_kernels_scalar_and_simd_agree(radii):
    # (the same operations in the same order, lane by lane: to the bit)
    from pysurfaceevolver import _core
    ev = _drop_among_flux_spheres(radii)
    saved = ev._save()
    found = []
    for simd in (True, False):
        was = _core.native_simd(simd)
        try:
            ev.restore(saved)
            ev.command("g 3; hessian")
            found.append((ev.energy, list(ev.bodies().volume), np.asarray(ev.vertices).copy()))
        finally:
            _core.native_simd(was)
    assert found[0][0] == found[1][0] and found[0][1] == found[1][1]
    assert np.array_equal(found[0][2], found[1][2])


def test_fast_gradient_steps_reach_the_stock_searchs_equilibrium():
    # (a topology descent's steps: no move at scale 0 where nothing changed,
    # the scale from a parabola with the energy's slope; off by default)
    from pysurfaceevolver import _core
    found = []
    for fast in (False, True):
        was = _core.set_fast_steps(fast)
        try:
            ev = _drop_under_a_sphere()
            ev.body(1).target = 1.3*ev.body(1).target
            r = ev.relax(tol=1e-9)
            exact = _cap_pressure(ev.body(1).volume, 90.0)
            found.append((r.converged, ev.energy, abs(ev.body(1).pressure/exact - 1)))
        finally:
            _core.set_fast_steps(was)
    # (an equilibrium as good: a fixed mesh has many, apart by vertices
    # sliding along the surface, at the mesh's error, 1e-5 in energy here)
    assert found[0][0] and found[1][0]
    assert found[1][1] == pytest.approx(found[0][1], rel=1e-4)
    assert found[1][2] < max(2*found[0][2], 1e-3)


def _two_drops_on_ball_flux():
    """Two drops on a flux-gauge sphere, at its poles: two bodies, each its own
    copy of the volume method."""
    cap = shapes.cap(0.5, 0.25, nr=4, nt=12)
    v = cap.vertices + [0, 0, 1.0]
    rim = np.asarray(cap.rims["rim"])
    v[rim] /= np.linalg.norm(v[rim], axis=1)[:, None]
    f = np.asarray(cap.faces)
    w = v*[1, 1, -1]                        # (the south drop, its facets turned)
    verts = np.r_[v, w]
    faces = np.r_[f, f[:, ::-1] + len(v)]
    ev = pyse.Evolver()
    ev.load_arrays(verts, faces,
                   constraints={"ball": C.sphere((0, 0, 0), 1.0, contact_angle=60.0, gauge="flux")},
                   vertex_constraints={"ball": np.r_[rim, rim + len(v)]},
                   bodies=[pyse.Body(volume=None, faces=range(len(f))),
                           pyse.Body(volume=None, faces=range(len(f), 2*len(f)))])
    return ev


def test_a_body_gone_and_another_made_since_a_snapshot_restore_whole(tmp_path):
    # (a dissolved body's volume quantity and method go to free slots, and the
    # next body's take them; a snapshot put back over that marked the new
    # volume deleted under its live method: a 16-bead pack's dumps declared
    # body_7_vol_meth, never wrote it, and didn't load, ERROR 3226)
    ev = _two_drops_on_ball_flux()
    ev.command("convert_to_quantities")
    ev.body(1).target = ev.body(1).volume
    ev.body(2).target = ev.body(2).volume
    volumes, energy = list(ev.bodies().volume), ev.energy
    saved = ev._save()
    assert type(saved).__name__ == "_Saved"
    ev.command("dissolve body[2]")
    ev.command("pyse_b := new_body; set facet frontbody pyse_b where frontbody == 0 "
               "and backbody == 0")
    assert len(ev.bodies().ids) == 2
    ev.restore(saved)
    assert list(ev.bodies().ids) == [1, 2]
    assert list(ev.bodies().volume) == pytest.approx(volumes, rel=1e-12)
    assert ev.energy == pytest.approx(energy, rel=1e-12)
    path = tmp_path / "back.fe"
    ev.dump(path)
    ev.command("g 3")
    assert list(ev.bodies().volume) == pytest.approx(volumes, rel=1e-6)
    again = pyse.Evolver()
    again.load(path)
    assert list(again.bodies().volume) == pytest.approx(volumes, rel=1e-12)
