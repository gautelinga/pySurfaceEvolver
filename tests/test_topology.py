"""Topology changes done with Evolver's own commands (see docs/PLAN.md,
"Future: topology changes")."""

import warnings

import numpy as np
import pytest

import pysurfaceevolver as pyse
from pysurfaceevolver import constraints as C, shapes


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
    assert len(_pieces(ev)) == 1 and list(ev.bodies().ids) == [1]
    f = np.asarray(ev.mesh().facets)
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    assert np.all(np.unique(e, axis=0, return_counts=True)[1] == 2)      # closed, manifold
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
