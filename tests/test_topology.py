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
    # (the joined surface may come to rest on the plane between them: wetted)
    kinds = [(e.kind, e.constraint) for e in fell[0].changes]
    assert len(fell) == 1 and kinds[0][0] == "contact lines"
    assert all(k == ("solid", 1) for k in kinds[1:])
    k = steps.index(fell[0])
    assert fell[0].volumes == pytest.approx([steps[k - 1].value + other], rel=1e-8)
    assert _loops(_border(ev)) == 1
    for s in steps[k:]:
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
